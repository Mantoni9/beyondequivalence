import logging
import os
import time
import traceback
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from LLMBase import LLMBase
from llm_tool import Tool
from prompt import Prompt

logger = logging.getLogger(__name__)


class LLMHuggingFace(LLMBase):
    """HuggingFace Transformers backend for local LLaMA 3 / similar models.

    Parameters
    ----------
    model_path:
        Local path to a HuggingFace-compatible model directory.
    device_map:
        Passed to ``from_pretrained``; ``'auto'`` distributes layers
        across all visible GPUs.
    dtype:
        Weight dtype; ``torch.bfloat16`` is recommended for LLaMA 3.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_path  # used by LLMBase token-set helpers

        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._initialize_positive_negative_tokens()

        load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
        load_in_8bit = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"

        # Flash Attention 2 requires both CUDA and the flash-attn package.
        try:
            import flash_attn  # noqa: F401
            use_flash_attn = torch.cuda.is_available()
        except ImportError:
            use_flash_attn = False
            logger.info("flash-attn not installed — using default attention implementation.")

        # attn_implementation selection:
        #
        # - Non-quantized CUDA models: "flash_attention_2"
        #     Uses the transformers FA2 wrapper for maximum control.
        #
        # - NF4 / 8-bit + CUDA models: "sdpa"
        #     transformers 5.5.x has a bug in prepare_fa_kwargs_from_position_ids:
        #     during autoregressive decoding the query length is 1, so
        #     position_ids[0].diff() is empty → cu_seq_lens_q has one element →
        #     cu_seq_lens_q.diff().max() raises "numel() == 0".
        #     PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
        #     dispatches to the same FA2 CUDA kernels internally
        #     (torch.backends.cuda.flash_sdp_enabled() is True by default),
        #     so throughput is identical without the broken transformers wrapper.
        quantized = load_in_4bit or load_in_8bit
        if use_flash_attn:
            if quantized:
                attn_kwargs = {"attn_implementation": "sdpa"}
                logger.info(
                    "Flash Attention 2 + NF4/8-bit triggers a transformers 5.5.x bug "
                    "(empty cu_seq_lens_q during decode). Using SDPA instead — "
                    "PyTorch dispatches to FA2 kernels internally."
                )
            else:
                attn_kwargs = {"attn_implementation": "flash_attention_2"}
                logger.info("Flash Attention 2 enabled.")
        else:
            attn_kwargs = {}

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print(
                f"[Model] BitsAndBytesConfig: load_in_4bit=True  quant_type=nf4  "
                f"compute_dtype=bfloat16  double_quant=True",
                flush=True,
            )
            logger.info(
                f"Loading model from {model_path} "
                f"(device_map={device_map}, load_in_4bit=NF4, compute_dtype=bfloat16, double_quant=True)"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                **attn_kwargs,
            )
        elif load_in_8bit:
            logger.info(
                f"Loading model from {model_path} "
                f"(device_map={device_map}, load_in_8bit=True via BitsAndBytesConfig)"
            )
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                **attn_kwargs,
            )
        else:
            if torch.cuda.is_available():
                logger.info(
                    f"Loading model from {model_path} "
                    f"(device_map=auto, dtype={dtype}, CUDA)"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    **attn_kwargs,
                )
            elif torch.backends.mps.is_available():
                # device_map="auto" is not supported on MPS; load on CPU then move.
                logger.info(
                    f"Loading model from {model_path} "
                    f"(dtype=bfloat16, MPS)"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map=None,
                    trust_remote_code=True,
                )
                self.model = self.model.to("mps")
            else:
                logger.info(
                    f"Loading model from {model_path} "
                    f"(dtype=float32, CPU)"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                )
        self.model.eval()
        # LLaMA 3.x ships a generation_config.json with temperature/top_p set.
        # Those flags are invalid for greedy decoding (do_sample=False) and cause
        # generate() to hang on CUDA. Clear them explicitly.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.do_sample = False

        # ---- Model diagnostics ----
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] Parameters:  {num_params / 1e9:.2f}B", flush=True)
        print(f"[Model] dtype:       {next(self.model.parameters()).dtype}", flush=True)
        if hasattr(self.model, "hf_device_map"):
            print(f"[Model] device_map:  {self.model.hf_device_map}", flush=True)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved  = torch.cuda.memory_reserved(i)  / 1024 ** 3
                print(
                    f"[Model] GPU {i} VRAM:  {allocated:.2f} GB allocated / "
                    f"{reserved:.2f} GB reserved",
                    flush=True,
                )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _first_device(self) -> torch.device:
        """Return the device of the first model parameter (used as input device)."""
        return next(self.model.parameters()).device

    def _apply_template(self, prompt):
        """Tokenize *prompt* with the model's chat template and move to device.

        Accepts either a :class:`Prompt` object (calls ``.to_messages()``) or a
        plain :class:`str` (wrapped as a single user message).

        Returns
        -------
        input_ids : torch.Tensor  shape (1, seq_len)
        attention_mask : torch.Tensor  shape (1, seq_len), all-ones (no padding)
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt.to_messages()
        result = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # transformers 5.x may return a BatchEncoding instead of a plain Tensor.
        if isinstance(result, torch.Tensor):
            input_ids = result.to(self._first_device())
        else:
            input_ids = result["input_ids"].to(self._first_device())
        # Build an explicit all-ones attention_mask (no padding in single-sequence encoding).
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    # ------------------------------------------------------------------ #
    #  LLMBase interface                                                  #
    # ------------------------------------------------------------------ #

    def _resolve_eos_ids(self):
        """Return (eos_token_ids, pad_token_id) as (list[int], int).

        LLaMA 3.x tokenizers expose ``eos_token_id`` as a *list* of multiple
        end-of-sequence token IDs (e.g. ``[128001, 128008, 128009]``).
        ``generate()`` accepts a list for ``eos_token_id`` but **requires a
        scalar int** for ``pad_token_id``; passing a list there triggers the
        internal ``torch.max()`` call that raises
        "max(): Expected reduction dim to be specified for input.numel() == 0".
        """
        raw = self.tokenizer.eos_token_id
        eos_ids: list = raw if isinstance(raw, list) else [raw]
        # Use the dedicated pad token when available; fall back to the first EOS.
        pad_id: int = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else eos_ids[0]
        )
        return eos_ids, pad_id

    def get_text_completion(
        self, prompts: List[Prompt], max_new_tokens: int = 50
    ) -> List[str]:
        eos_ids, pad_id = self._resolve_eos_ids()
        completions: List[str] = []
        print(f"[get_text_completion] START batch_size={len(prompts)}", flush=True)
        t_batch = time.time()
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            try:
                t_tpl = time.time()
                input_ids, attention_mask = self._apply_template(prompt)
                print(f"[get_text_completion]   [{i+1}/{len(prompts)}] template:  "
                      f"{time.time()-t_tpl:.2f}s  (input_len={input_ids.shape[1]})", flush=True)

                n_input = input_ids.shape[1]
                t_gen = time.time()
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=eos_ids,
                    )
                new_tokens = output_ids[0, n_input:]
                print(f"[get_text_completion]   [{i+1}/{len(prompts)}] generate:  "
                      f"{time.time()-t_gen:.2f}s  (new_tokens={new_tokens.shape[0]})", flush=True)

                t_dec = time.time()
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"[get_text_completion]   [{i+1}/{len(prompts)}] decode:    "
                      f"{time.time()-t_dec:.2f}s", flush=True)
                completions.append(text)
            except Exception as e:
                logger.error(
                    "Error in get_text_completion: %s\n%s",
                    e,
                    traceback.format_exc(),
                )
                completions.append("")
            print(f"[get_text_completion] prompt {i+1}/{len(prompts)} done in "
                  f"{time.time()-t0:.1f}s", flush=True)
        print(f"[get_text_completion] DONE batch in {time.time()-t_batch:.1f}s", flush=True)
        return completions

    def get_confidence_first_token(self, prompts: List[Prompt]) -> List[float]:
        """Return P(yes) / (P(yes) + P(no)) from the next-token logits.

        Uses a single forward pass (no generation) to obtain the full
        next-token distribution, then aggregates probabilities over all
        vocabulary tokens that match the positive / negative patterns
        defined by :meth:`LLMBase._initialize_positive_negative_tokens`.
        """
        scores: List[float] = []
        print(f"[get_confidence_first_token] START batch_size={len(prompts)}", flush=True)
        t_batch = time.time()
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            try:
                t_tpl = time.time()
                input_ids, attention_mask = self._apply_template(prompt)
                print(f"[get_confidence_first_token]   [{i+1}/{len(prompts)}] template: "
                      f"{time.time()-t_tpl:.2f}s  (input_len={input_ids.shape[1]})", flush=True)

                t_fwd = time.time()
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                print(f"[get_confidence_first_token]   [{i+1}/{len(prompts)}] forward:  "
                      f"{time.time()-t_fwd:.2f}s", flush=True)

                # logits: (1, seq_len, vocab_size) → last position
                next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)
                probs = torch.softmax(next_token_logits, dim=-1).cpu()
                # Guard against token IDs that exceed the model's actual output
                # vocab size (can differ from tokenizer.vocab_size when the
                # embedding table is padded by bitsandbytes quantization).
                vocab_size = probs.shape[0]

                yes_prob = max(
                    (probs[tid].item() for tid in self.positive_token_ids if tid < vocab_size),
                    default=0.0,
                )
                no_prob = max(
                    (probs[tid].item() for tid in self.negative_token_ids if tid < vocab_size),
                    default=0.0,
                )

                total = yes_prob + no_prob
                score = yes_prob / total if total > 0 else 0.5
                scores.append(score)
                print(f"[get_confidence_first_token]   [{i+1}/{len(prompts)}] score:    "
                      f"{score:.4f}  (yes={yes_prob:.4f}, no={no_prob:.4f})", flush=True)
            except Exception as e:
                logger.error(
                    "Error in get_confidence_first_token: %s\n%s",
                    e,
                    traceback.format_exc(),
                )
                scores.append(0.5)
            print(f"[get_confidence_first_token] prompt {i+1}/{len(prompts)} done in "
                  f"{time.time()-t0:.1f}s", flush=True)
        print(f"[get_confidence_first_token] DONE batch in {time.time()-t_batch:.1f}s", flush=True)
        return scores

    def get_confidence_with_tools(
        self,
        prompts: List[Prompt],
        tools: Sequence[Tool],
        max_iterations: int = 10,
    ) -> List[float]:
        raise NotImplementedError(
            "Tool calling is not supported for the local HuggingFace backend. "
            "Use LLMOpenAI with a vLLM / compatible endpoint instead."
        )
