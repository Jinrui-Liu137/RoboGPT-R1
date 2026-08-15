# -*- coding: utf-8 -*-

import json, os, re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------- Logging setup ----------
def setup_logging():
    """Configure logging."""
    log_file = os.environ.get("REWARD_LOG_PATH", "./reward_evaluation.log")
    
    # Create the logger.
    logger = logging.getLogger('reward_function')
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times.
    if not logger.handlers:
        # File handler.
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter.
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ---------- Fallback action map ----------
def _get_fallback_action_map() -> Dict[int, str]:
    """
    Fallback when action_map cannot be extracted from the prompt.
    It can return a generic action_map or load one from a file.
    """
    fallback_map = {}
    
    # Option 3: try loading from an environment variable (retain the original fallback logic).
    action_map_path = os.environ.get("ACTION_MAP_PATH", "").strip()
    if action_map_path and os.path.exists(action_map_path):
        try:
            with open(action_map_path, "r", encoding="utf-8") as f:
                txt = f.read()
            amap: Dict[int, str] = {}
            for m in _ACTION_LINE_RE.finditer(txt):
                aid = int(m.group(1))
                name = m.group(2).strip()
                if aid not in amap:
                    amap[aid] = name
            logger.info(f"Loaded {len(amap)} action mappings from the fallback path")
            return amap
        except Exception as e:
            logger.error(f"Failed to load fallback action_map: {e}")
    
    return fallback_map

_ACTION_LINE_RE = re.compile(
    r"action\s*id\s*(\d+)\s*:\s*([A-Za-z0-9_\-./() ]+?)(?=,|;|\n|$)",
    re.IGNORECASE,
)

def _extract_action_map_from_prompt(prompt_text: str) -> Dict[int, str]:
    """
    Dynamically extract action_map from prompt_text.
    """
    if not isinstance(prompt_text, str):
        return {}
    
    amap: Dict[int, str] = {}
    for m in _ACTION_LINE_RE.finditer(prompt_text):
        aid = int(m.group(1))
        name = m.group(2).strip()
        if aid not in amap:
            amap[aid] = name
    
    return amap

def _build_name2id_map(action_map: Dict[int, str]) -> Dict[str, int]:
    """
    Build a reverse lookup table from action names to IDs.
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""
    
    return {_norm(v): k for k, v in action_map.items()}

# ---------- Action-name parsing regex ----------
_STEP_NAME_RE = re.compile(
    r"\b(goto|pickup|put|toggle|cool|slice|clean|heat)\s+([a-z0-9_]+)\b",
    re.IGNORECASE,
)

# ---------- Basic utilities ----------
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CODE_JSON_RE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _safe_json(s: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    m = _ANSWER_RE.search(t) or _CODE_JSON_RE.search(t) or _FIRST_OBJ_RE.search(t)
    if m:
        t = m.group(1) if m.re is _CODE_JSON_RE else (m.group(1) if m.re is _ANSWER_RE else m.group(0))
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _extract_plan(obj: Optional[Dict[str, Any]]) -> List[Any]:
    if not isinstance(obj, dict): 
        return []
    p = obj.get("executable_plan", [])
    return p if isinstance(p, list) else []

def _parse_step(step: Any, action_map: Dict[int, str], name2id: Dict[str, int]) -> Tuple[Optional[int], Optional[str]]:
    """
    Normalize a step to (action_id, action_name):
    - dict: read id/name; infer a missing id from name
    - str: parse "verb target" and infer id
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""
    
    if isinstance(step, dict):
        aid = step.get("action_id")
        an = step.get("action_name")
        if isinstance(an, str) and an.strip():
            an_norm = _norm(an)
            if not isinstance(aid, int) and an_norm in name2id:
                aid = name2id[an_norm]
            # Return the normalized name.
            if isinstance(aid, int) and aid in action_map:
                return aid, action_map[aid]
            return aid if isinstance(aid, int) else None, an if isinstance(an, str) else None
        # No action_name: use the ID only.
        if isinstance(aid, int) and aid in action_map:
            return aid, action_map[aid]
        return None, None

    if isinstance(step, str):
        s_norm = _norm(step)
        # 1) Match a complete action name directly.
        if s_norm in name2id:
            aid = name2id[s_norm]
            return aid, action_map[aid]
        # 2) Parse the "verb target" pattern, then convert to a canonical name.
        m = _STEP_NAME_RE.search(s_norm)
        if m:
            name = f"{m.group(1).lower()} {m.group(2).lower()}"
            if name in name2id:
                aid = name2id[name]
                return aid, action_map[aid]
        return None, step  # At minimum, return the original string as a name for reference.
    return None, None

def _names_seq(plan: List[Any], action_map: Dict[int, str], name2id: Dict[str, int]) -> List[str]:
    """
    Extract the sequence of action names from a plan.
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""
    
    out: List[str] = []
    for step in plan:
        aid, an = _parse_step(step, action_map, name2id)
        if isinstance(an, str) and an.strip():
            out.append(_norm(an))
    return out

def _lcs_len(a: List[str], b: List[str]) -> int:
    """
    Calculate the longest common subsequence length.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0: 
        return 0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            dp[i][j] = dp[i-1][j-1]+1 if ai==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def _nlcs(a: List[str], b: List[str]) -> float:
    """
    Normalized longest common subsequence similarity.
    """
    if not a and not b: 
        return 1.0
    den = max(len(a), len(b))
    return (_lcs_len(a, b) / den) if den else 0.0

# ---------- Format scoring components ----------
def _fmt_top_fields(o: Optional[Dict[str, Any]]) -> float:
    """
    Check the format completeness of top-level fields.
    """
    need = {
        "visual_state_description": str, 
        "reasoning_and_reflection": str, 
        "language_plan": str, 
        "executable_plan": list
    }
    if not isinstance(o, dict): 
        return 0.0
    ok = sum(1 for k, t in need.items() if isinstance(o.get(k), t))
    return ok / len(need)

def _fmt_plan_shape(plan: List[Any], action_map: Dict[int, str], name2id: Dict[str, int]) -> float:
    """
    Check the format validity of each action in the plan.
    """
    if not plan: 
        return 0.0
    ok = 0
    max_id = max(action_map.keys()) if action_map else 224
    for step in plan:
        aid, an = _parse_step(step, action_map, name2id)
        if isinstance(aid, int) and 0 <= aid <= max_id and isinstance(an, str) and an.strip():
            ok += 1
    return ok / len(plan)

def _fmt_id_name_match(plan: List[Any], action_map: Dict[int, str], name2id: Dict[str, int]) -> float:
    """
    Check whether action IDs match action names.
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""
    
    if not plan or not action_map: 
        return 0.0
    cnt = cor = 0
    for step in plan:
        aid, an = _parse_step(step, action_map, name2id)
        if isinstance(aid, int) and aid in action_map and isinstance(an, str):
            cnt += 1
            cor += 1 if _norm(an) == _norm(action_map[aid]) else 0
    return (cor / cnt) if cnt else 0.0

def _extract_images_info(ri: Dict[str, Any]) -> str:
    """Extract image information."""
    images = ri.get("images", [])
    if not images:
        return "No images"
    
    if isinstance(images, str):
        return f"Single image: {images}"
    elif isinstance(images, list):
        return f"{len(images)} images: {', '.join(images[:3])}" + ("..." if len(images) > 3 else "")
    else:
        return f"Image information: {type(images)}"

def _log_detailed_evaluation(
    sample_idx: int,
    ri: Dict[str, Any], 
    action_map: Dict[int, str],
    pred: Optional[Dict[str, Any]],
    gt: Optional[Dict[str, Any]],
    pplan: List[Any],
    gplan: List[Any],
    pnames: List[str],
    gnames: List[str],
    scores: Dict[str, float]
):
    """Log detailed evaluation information."""
    
    # Basic information.
    logger.info(f"\n" + "="*80)
    logger.info(f"Evaluation details for sample #{sample_idx}")
    logger.info(f"="*80)
    
    # Action map information.
    if action_map:
        first_action = f"ID {min(action_map.keys())}: {action_map[min(action_map.keys())]}"
        last_action = f"ID {max(action_map.keys())}: {action_map[max(action_map.keys())]}"
        logger.info(f"Action map: {len(action_map)} actions")
        logger.info(f"First action: {first_action}")
        logger.info(f"Last action: {last_action}")
    else:
        logger.warning("Action map: no action mappings extracted")
    
    # Image information.
    images_info = _extract_images_info(ri)
    logger.info(f"Image information: {images_info}")
    
    # Model response.
    model_response = ri.get("response", "")[:50000]  # Limit length.
    logger.info(f"Model response (first 500 characters): {model_response}")
    if len(ri.get("response", "")) > 50000:
        logger.info(f"... (total length: {len(ri.get('response', ''))} characters)")
    
    # Ground truth.
    ground_truth = ri.get("ground_truth", "")[:50000]  # Limit length.
    logger.info(f"Ground truth (first 500 characters): {ground_truth}")
    if len(ri.get("ground_truth", "")) > 50000:
        logger.info(f"... (total length: {len(ri.get('ground_truth', ''))} characters)")
    
    # Parsing results.
    logger.info(f"Predicted plan parsing: {'success' if pred else 'failed'}")
    logger.info(f"Ground-truth plan parsing: {'success' if gt else 'failed'}")
    
    if pplan:
        logger.info(f"Predicted action sequence ({len(pplan)} steps): {pnames[:5]}" + ("..." if len(pnames) > 5 else ""))
    else:
        logger.info("Predicted action sequence: no valid actions")
    
    if gplan:
        logger.info(f"Ground-truth action sequence ({len(gplan)} steps): {gnames[:5]}" + ("..." if len(gnames) > 5 else ""))
    else:
        logger.info("Ground-truth action sequence: no valid actions")
    
    # Score details.
    logger.info(f"Format score: {scores['format']:.4f}")
    logger.info(f"Accuracy score: {scores['accuracy']:.4f}")
    logger.info(f"Overall score: {scores['overall']:.4f}")
    
    # LCS matching details.
    if pnames and gnames:
        lcs_length = _lcs_len(pnames, gnames)
        logger.info(f"LCS match length: {lcs_length}/{max(len(pnames), len(gnames))}")
        
        # Identify matching and non-matching actions.
        matched_actions = []
        i, j = 0, 0
        while i < len(pnames) and j < len(gnames):
            if pnames[i] == gnames[j]:
                matched_actions.append(pnames[i])
                i += 1
                j += 1
            elif i < len(pnames) - 1 and pnames[i + 1] == gnames[j]:
                i += 1
            elif j < len(gnames) - 1 and pnames[i] == gnames[j + 1]:
                j += 1
            else:
                i += 1
                j += 1
        
        if matched_actions:
            logger.info(f"Matching actions: {matched_actions[:3]}" + ("..." if len(matched_actions) > 3 else ""))
    
    logger.info(f"="*80)

# ---------- Main entry point ----------
def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
    """
    Main function for calculating reward scores.
    Dynamically extract action_map from the content field.
    
    Args:
        reward_inputs: Input list containing fields such as response, ground_truth, and content.
        format_weight: Weight of the format score; defaults to 0.2.
        
    Returns:
        A list of dictionaries containing overall, format, and accuracy scores.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch`.")
    
    logger.info(f"Starting evaluation of {len(reward_inputs)} samples")
    
    # DEBUG: log the structure of reward_inputs (on the first run).
    if reward_inputs and isinstance(reward_inputs[0], dict):
        logger.info(f"reward_inputs fields: {list(reward_inputs[0].keys())}")
    
    fw = max(0.0, min(1.0, float(format_weight)))
    out: List[Dict[str, float]] = []

    for idx, ri in enumerate(reward_inputs):
        # Extract action_map from the content field.
        content = ri.get("content", "")
        action_map = {}
        
        if content and isinstance(content, str):
            action_map = _extract_action_map_from_prompt(content)
            if action_map:
                logger.debug(f"Sample #{idx}: extracted {len(action_map)} actions from content; range: [{min(action_map.keys())}, {max(action_map.keys())}]")
        
        # Use the fallback if action_map was not extracted.
        if not action_map:
            action_map = _get_fallback_action_map()
            if action_map:
                logger.warning(f"Sample #{idx}: could not extract action_map from content; using fallback mapping")
            else:
                logger.error(f"Sample #{idx}: could not obtain action_map")
                out.append({"overall": 0.0, "format": 0.0, "accuracy": 0.0})
                continue
        
        # Build the reverse lookup table.
        name2id = _build_name2id_map(action_map)
        
        # Parse the prediction and ground truth.
        pred = _safe_json(ri.get("response", ""))
        gt = _safe_json(ri.get("ground_truth", ""))

        pplan = _extract_plan(pred)
        gplan = _extract_plan(gt)
        pnames = _names_seq(pplan, action_map, name2id)
        gnames = _names_seq(gplan, action_map, name2id)

        # Calculate the format score.
        f1 = _fmt_top_fields(pred)
        f2 = _fmt_plan_shape(pplan, action_map, name2id)
        f3 = _fmt_id_name_match(pplan, action_map, name2id)
        fmt = f1 * 0.3 + f2 * 0.3 + f3 * 0.4

        # Calculate the accuracy score (based on LCS similarity).
        acc = _nlcs(pnames, gnames)
        
        # Overall score.
        overall = (1.0 - fw) * acc + fw * fmt

        scores = {
            "overall": float(overall), 
            "format": float(fmt), 
            "accuracy": float(acc)
        }
        out.append(scores)
        
        # Detailed logging.
        _log_detailed_evaluation(
            idx, ri, action_map, pred, gt, 
            pplan, gplan, pnames, gnames, scores
        )
    
    # Batch summary.
    avg_overall = sum(s["overall"] for s in out) / len(out)
    avg_format = sum(s["format"] for s in out) / len(out)
    avg_accuracy = sum(s["accuracy"] for s in out) / len(out)
    
    logger.info("\nBatch evaluation summary:")
    logger.info(f"Sample count: {len(out)}")
    logger.info(f"Average overall score: {avg_overall:.4f}")
    logger.info(f"Average format score: {avg_format:.4f}")
    logger.info(f"Average accuracy score: {avg_accuracy:.4f}")
    
    return out
