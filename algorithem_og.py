"""
Hebrew Entity Resolution v7 (Best / v9)
Framework v7 compliant single-script entity resolution for Hebrew names.

Resolves name mentions from call data into canonical entities using:
1. Mention explosion (split by "+", assign verified)
2. Normalization (unicode, whitespace, punctuation)
3. Vectorization (TF-IDF character n-grams + tokens)
4. Graph building (per-phone similarity graphs)
5. Clustering (HAC with hard constraints)
6. Resolution (priority cascade labeling)
6.25. Bridge Merging (Cube2-driven safe linking)  <-- THE KEY UPGRADE
6.5. Cluster Merging (Canonical similarity)
7. Global Cross-Phone Linking (DSU with constraints)
8. Entity Fusion (Unified global identity)

Robustness features:
- Two-layer bridge prevention (similarity cap + AmbiguityGate)
- Verified constraints (hard-block + anchor by phone,verified_name)
- Call-level nickname matching (PhoneAliasIndex)
- Mention validity scoring (filler hygiene)
- Deterministic clustering
"""

import re
import json
import logging
import random
import unicodedata
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def _lcs_ratio(s1: str, s2: str) -> float:
    """LCS-based normalized similarity (0-100 scale).
    Matches rapidfuzz.fuzz.ratio exactly (both use Indel/LCS distance).
    Uses O(min(n,m)) space via row-swap DP."""
    if not s1 and not s2:
        return 100.0
    if not s1 or not s2:
        return 0.0
    # Ensure s2 is the shorter string to minimize memory
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    n, m = len(s1), len(s2)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return 2.0 * prev[m] / (n + m) * 100.0


def _char_ratio(s1: str, s2: str) -> float:
    """Character-level similarity (0-100 scale).
    Uses rapidfuzz when available, LCS DP otherwise."""
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2)
    return _lcs_ratio(s1, s2)


def _token_set_ratio(s1: str, s2: str) -> float:
    """Token set similarity (0-100 scale).
    Compares sorted intersection/remainder combinations using LCS ratio."""
    if HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(s1, s2)
    t1, t2 = set(s1.split()), set(s2.split())
    inter = sorted(t1 & t2)
    d1, d2 = sorted(t1 - t2), sorted(t2 - t1)
    inter_s = ' '.join(inter)
    c1 = (inter_s + ' ' + ' '.join(d1)).strip()
    c2 = (inter_s + ' ' + ' '.join(d2)).strip()
    candidates = []
    if inter_s and c1:
        candidates.append(_lcs_ratio(inter_s, c1))
    if inter_s and c2:
        candidates.append(_lcs_ratio(inter_s, c2))
    if c1 and c2:
        candidates.append(_lcs_ratio(c1, c2))
    return max(candidates) if candidates else 0.0


def _token_sort_ratio(s1: str, s2: str) -> float:
    """Token sort similarity (0-100 scale).
    Sorts tokens alphabetically then compares using LCS ratio."""
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(s1, s2)
    sorted1 = ' '.join(sorted(s1.split()))
    sorted2 = ' '.join(sorted(s2.split()))
    return _lcs_ratio(sorted1, sorted2)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Centralized configuration for entity resolution."""

    # Cube names (for framework integration)
    CUBE_CALLS: str = 'cube1'
    CUBE_CONTACTS: str = 'cube2'

    # Column mappings for cube1 (calls)
    COL_CALL_ID: str = 'call_id'
    COL_DATE: str = 'date'
    COL_PHONE_A: str = 'pstn_A'
    COL_PHONE_B: str = 'pstn_B'
    COL_CLEAN_NAME_A: str = 'clean_name_A'
    COL_CLEAN_NAME_B: str = 'clean_name_B'
    COL_VERIFIED_NAME_A: str = 'verified_name_A'
    COL_VERIFIED_NAME_B: str = 'verified_name_B'
    COL_VERIFIED_NICKNAMES_A: str = 'verified_nicknames_A'
    COL_VERIFIED_NICKNAMES_B: str = 'verified_nicknames_B'
    COL_VERIFIED_ENTITY_ID_A: str = 'verified_entity_id_A'
    COL_VERIFIED_ENTITY_ID_B: str = 'verified_entity_id_B'
    COL_VERIFIED_STATUS_A: str = 'verified_status_A'
    COL_VERIFIED_STATUS_B: str = 'verified_status_B'
    COL_VERIFIED_ID_NUMBER_A: str = 'verified_id_number_A'
    COL_VERIFIED_ID_NUMBER_B: str = 'verified_id_number_B'

    # Column mappings for cube2 (contacts)
    COL_CONTACT_PHONE: str = 'phone_number'
    COL_CONTACT_NAME: str = '~parent_person_title'
    COL_CONTACT_NICKNAME: str = '~parent_person_nicknames'
    COL_CONTACT_ENTITY_ID: str = '~parent_person_id'
    COL_CONTACT_STATUS: str = 'status'
    COL_CONTACT_ID_NUMBER: str = 'id_number'

    # Similarity weights (must sum to 1.0)
    WEIGHT_CHAR_NGRAM: float = 0.40
    WEIGHT_TOKEN_SET: float = 0.35
    WEIGHT_IDF_JACCARD: float = 0.25

    # Thresholds
    SIMILARITY_THRESHOLD: float = 0.70
    # Cross-phone linking (phone is a soft prior, not a hard boundary)
    DEBUG_CROSS_PHONE: bool = False
    DEBUG_TOKEN_MATCHING: bool = False
    DEBUG_TOKEN_MATCHING_THRESHOLD: int = 70  # Log near-misses above this threshold
    CROSS_PHONE_ANCHOR_SORT_THRESHOLD: int = 85
    # Fix 28.2b: Corrected containment thresholds - anchors should be MORE LENIENT (lower threshold)
    # Previously these were backwards (anchor=90, unverified=85), contradicting the design principle
    # stated in variant_aware_cluster_score that anchors get "more lenient" treatment.
    CROSS_PHONE_ANCHOR_CONTAINMENT_SET_THRESHOLD: int = 85  # More lenient for verified anchors
    CROSS_PHONE_ANCHOR_LAST_TOKEN_FUZZY_THRESHOLD: int = 85  # For anchor fuzzy last-token match
    CROSS_PHONE_UNVERIFIED_SORT_THRESHOLD: int = 92
    CROSS_PHONE_UNVERIFIED_CONTAINMENT_SET_THRESHOLD: int = 90  # Stricter for unverified
    CROSS_PHONE_UNVERIFIED_LAST_TOKEN_FUZZY_THRESHOLD: int = 93  # For unverified fuzzy last-token
    CROSS_PHONE_FIRST_LAST_FUZZY_THRESHOLD: int = 90  # For general first/last fuzzy match
    CLUSTER_MERGE_SORT_THRESHOLD: int = 88  # For cluster merge sort ratio
    CLUSTER_MERGE_FIRST_LAST_THRESHOLD: int = 85  # For cluster merge first/last match
    # Fix 28.7 (Finding B): Align containment threshold with variant_aware_cluster_score
    CLUSTER_MERGE_CONTAINMENT_SET_THRESHOLD: int = 90  # For cluster merge containment check
    CUBE2_MATCH_THRESHOLD: float = 0.75
    CUBE2_MARGIN_THRESHOLD: float = 0.15
    # Fix 29.9: Relaxed margin for global matches with entity_id.
    # At scale (400+ records), many similar Arabic names compress margin below 0.15
    # even for correct matches. Entity_id already confirms identity, so margin
    # protection against ambiguity is redundant.
    CUBE2_MARGIN_THRESHOLD_WITH_ENTITY_ID: float = 0.05
    # Fix 29.7: Configurable global match thresholds (previously hardcoded in match_global)
    GLOBAL_MATCH_THRESHOLD: float = 0.90
    GLOBAL_MATCH_NO_ENTITY_THRESHOLD: float = 0.95
    VERIFIED_MATCH_THRESHOLD: float = 0.50
    # Kunya ambiguity hard guard for singleton / kunya-only evidence.
    # Prevents hard entity_id assignment when a kunya is shared by multiple
    # entities in the global phonebook unless stronger corroboration exists.
    KUNYA_AMBIGUITY_HARD_GUARD: bool = True
    KUNYA_GUARD_MULTI_MENTION_MIN_MARGIN: float = 0.20
    KUNYA_GUARD_MULTI_MENTION_MIN_SCORE: float = 0.90
    # When margin is extremely high, the scoring itself has fully disambiguated
    # the kunya — even a single mention is safe to assign.
    KUNYA_GUARD_SINGLE_HIGH_MARGIN: float = 0.80
    KUNYA_GUARD_SINGLE_HIGH_SCORE: float = 0.95
    # Singleton context guard for negation/list/repetition ambiguity.
    SINGLETON_CONTEXT_HARD_GUARD: bool = True
    SINGLETON_REPETITION_MIN_COUNT: int = 3
    SINGLETON_LIST_MIN_UNIQUE_TOKENS: int = 3
    SINGLETON_LOW_MARGIN_THRESHOLD: float = 0.08
    SINGLETON_LOW_TOP_SCORE_THRESHOLD: float = 0.90
    # Controlled same-phone alias propagation for unresolved kunya-only clusters.
    KUNYA_ALIAS_PROPAGATION_ENABLED: bool = True
    KUNYA_ALIAS_PROPAGATION_MIN_SCORE: float = 0.85
    KUNYA_ALIAS_PROPAGATION_MIN_MARGIN: float = 0.12
    # Fix 29.10: Configurable IDF demotion threshold (previously hardcoded 1.5).
    # Contacts with mean_idf below this value are demoted HIGH→MED to prevent
    # "black hole" merges from very common names. At larger scale the corpus
    # shifts IDF values — this knob allows tuning without code changes.
    IDF_DEMOTION_THRESHOLD: float = 1.5
    # Fix 29.23: Minimum corpus size before IDF demotion is applied.
    # For small corpora (< 50 mentions), log(N/df) produces artificially low IDF
    # values even for rare names. Skip IDF demotion entirely for small corpora.
    MIN_CORPUS_FOR_IDF_DEMOTION: int = 50

    # ------------------------------------------------------------------
    # cube2 (Phonebook) as first-class evidence (v9)
    # ------------------------------------------------------------------
    # Contact-name quality gating is intentionally conservative for Hebrew ER:
    # - Generic labels like 'אמא', 'עבודה', 'מונית' should NOT become strong anchors.
    # - Two-token real names (even if common) like 'יוסי כהן' should be allowed.
    CUBE2_PHONEBOOK_MIN_TOKENS: int = 2
    CUBE2_PHONEBOOK_MIN_NON_GENERIC_TOKENS: int = 2
    # Tiers: LOW / MED / HIGH. Used for (a) PHONEBOOK resolution eligibility,
    # (b) intra-phone cube2 bridging, and (c) cross-phone anchor behavior.
    CUBE2_PHONEBOOK_MIN_QUALITY_TIER_FOR_RESOLUTION: str = 'HIGH'
    CUBE2_PHONEBOOK_STRONG_QUALITY_TIERS: Tuple[str, ...] = ('HIGH',)
    # Generic / role-like tokens commonly found in phonebooks (not person identifiers).
    # Normalization is applied before comparison, so include plain forms here.
    CUBE2_GENERIC_TOKENS: Set[str] = field(default_factory=lambda: {
        # Family roles
        'אמא', 'אבא', 'אח', 'אחות', 'אחי', 'אחותי', 'בן', 'בת', 'דוד', 'דודה', 'סבא', 'סבתא',
        # Common labels / places
        'בית', 'עבודה', 'משרד', 'מזכירה', 'מזכירות', 'מוקד', 'שירות', 'קבלה',
        # Services / professions
        'מונית', 'טקסי', 'נהג', 'נהגת', 'רופא', 'דוקטור', 'מרפאה', 'טכנאי', 'שרברב', 'השרברב',
        'חשמלאי', 'אינסטלטור', 'מוסך', 'פנצ\'ריה',
        # Generic relationship labels
        'חבר', 'חברה', 'שכן', 'שכנה',
        # English fallbacks
        'mom', 'dad', 'home', 'work', 'office', 'taxi',
    })

    # Stage 6.25: cube2-driven intra-phone bridging (cluster-level, constraint-aware)
    CUBE2_BRIDGE_ENABLED: bool = True
    CUBE2_BRIDGE_MIN_SCORE: float = 0.75
    CUBE2_BRIDGE_MIN_MARGIN: float = 0.20
    # If a phone has many contacts, we assume multi-identity risk and require entity_id for bridging.
    CUBE2_BRIDGE_MAX_CONTACTS_PER_PHONE: int = 25
    # Safety cap: do not merge huge components via a single contact-name key unless entity_id exists.
    CUBE2_BRIDGE_MAX_CLUSTERS_PER_CONTACT: int = 8
    CUBE2_BRIDGE_REQUIRE_ENTITY_ID_WHEN_MANY_CONTACTS: bool = True

    # Clustering parameters
    CW_MAX_ITERATIONS: int = 20
    CW_SEED: int = 42

    # Special tokens and patterns
    BLMZ_TOKENS: Tuple[str, ...] = ('בלמ״ז', 'בלמז', 'דובר לא מזוהה', 'לא מזוהה')

    # --- Yanis API (Name→Name, batch, 20yr, highly trusted) ---
    API_ENABLED: bool = False               # Master kill switch
    API_A_URL: str = ''                     # Yanis endpoint
    API_A_TOKEN: str = ''                   # Authorization header value
    API_TIMEOUT: int = 30                   # Seconds per batch call
    API_BATCH_MAX_NAMES: int = 2000         # Safety cap on batch size

    # Veto/Confirm (edges 0.70-0.90)
    API_VETO_ENABLED: bool = True
    API_VETO_SCORE_MIN: float = 0.70        # Must match SIMILARITY_THRESHOLD
    API_VETO_SCORE_MAX: float = 0.90
    API_VETO_THRESHOLD: float = 60.0        # API < this → block merge
    API_CONFIRM_THRESHOLD: float = 80.0     # API ≥ this → tag confirmed

    # Rescue (pairs 0.45-0.69, below edge threshold)
    API_RESCUE_ENABLED: bool = True
    API_RESCUE_SCORE_MIN: float = 0.45
    API_RESCUE_SCORE_MAX: float = 0.69
    API_RESCUE_THRESHOLD: float = 85.0
    API_RESCUE_MIN_TOKENS: int = 2          # Single-token names too ambiguous
    API_RESCUE_ALLOW_EID_PROPAGATION: bool = False
    API_RESCUE_BYPASS_COHESION_SINGLETONS: bool = True

    # Cohesion bypass for high-confidence API confirmation (Phase 2)
    API_CONFIRM_BYPASS_COHESION: bool = True
    API_CONFIRM_BYPASS_THRESHOLD: float = 90.0   # API score ≥ this → skip cohesion

    # Discovery (Phase 4: let Yanis find matches the fuzzy scorer missed)
    API_DISCOVERY_ENABLED: bool = True
    API_DISCOVERY_THRESHOLD: float = 75.0   # API score >= this -> merge

    # Stage 6.2: Verified-Anchor API Attachment (Fix 30.1)
    # Injects entity_id onto unresolved clusters BEFORE Stage 7, so that
    # Phase 0 hard-links bypass the cohesion gate entirely.
    API_ANCHOR_ATTACH_ENABLED: bool = True
    API_ANCHOR_ATTACH_STRONG_MIN_SCORE: int = 60    # minScore for batch API call
    API_ANCHOR_ATTACH_WEAK_MIN_SCORE: int = 75      # For weak-evidence corroboration
    API_ANCHOR_ATTACH_MAX_CANDIDATES: int = 10       # Anchors per unresolved cluster
    API_ANCHOR_ATTACH_CONFIRM_THRESHOLD: float = 96.0
    API_ANCHOR_ATTACH_DUAL_CONFIRM: float = 93.0
    API_ANCHOR_ATTACH_DUAL_MIN_SUPPORT: int = 2
    API_ANCHOR_ATTACH_WEAK_CONFIRM: float = 97.0    # strong_to_weak with entity_id
    API_ANCHOR_ATTACH_VETO_THRESHOLD: float = 60.0
    API_ANCHOR_ATTACH_MARGIN_MIN: float = 5.0        # Gap between top-2 different-eid anchors

    # Fix 31.1: Confusable Given Name Safety Gate
    CONFUSABLE_NAME_GUARD_ENABLED: bool = True
    CONFUSABLE_GIVEN_NAME_MIN_SIMILARITY: int = 74   # Floor for "looks similar" (Fix 31.3: lowered from 80 to catch מחמד/אחמד at 75%)
    CONFUSABLE_GIVEN_NAME_MAX_IDENTITY: int = 97      # Ceiling — above this = same name
    CONFUSABLE_FAMILY_MATCH_MIN_SIMILARITY: int = 80  # Same family overrides confusability
    API_ANCHOR_ATTACH_CONFUSABLE_THRESHOLD: float = 99.0  # Virtually blocks attachment

    # Stage 7: Bundle-level scoring (replaces _api_representative_name)
    API_BUNDLE_SCORING_ENABLED: bool = True

    # Noise detection (structural, no API)
    NOISE_DETECTION_ENABLED: bool = True

    # Noise tokens (Hebrew punctuation versions - will be normalized on access)
    NOISE_TOKENS: Set[str] = field(default_factory=lambda: {
        # Military ranks/abbreviations
        # Note: 'בר' removed - it's a common Hebrew first name (Bar), not just rabbinical connector
        # Keep 'ב״ר' (with gershayim) for "Ben Rav" abbreviation
        'מ״פ', 'מפ', 'ב״ר', 'ככה״נ', 'ככהנ',
        'סגן', 'רס״ן', 'רסן', 'סא״ל', 'סאל',
        # System annotations (auto-identification markers)
        # Note: 'לא' omitted as it's too common in Hebrew and could filter real name parts
        'מזוהה', 'אוטומטית', 'זוהה', 'ידוע',
    })

    # Fix 29.37b: Common Arabic given names (Hebrew transliteration)
    # Used by _detect_nickname_hijack to distinguish given names from family names
    # in 2-token kunya patterns: "אבו-X Y" where Y could be given name OR family name.
    # Examples:
    #   "אבו-אנס מחמד" → "מחמד" IS a given name → skip (no family to extract)
    #   "אבו-אנס ברביר" → "ברביר" is NOT a given name → extract as family
    COMMON_ARABIC_GIVEN_NAMES: Set[str] = field(default_factory=lambda: {
        # Very common male names (top 20)
        'מחמד', 'מוחמד', 'אחמד', 'עלי', 'חסן', 'חוסין', 'עמר', 'סאלם', 'יוסף',
        'אברהים', 'אברהם', 'חאלד', 'ראמי', 'סמי', 'נאסר', 'פאדי', 'באסם',
        'גמאל', "ג'מאל", 'טארק', 'מוסטפא', 'ריאד', 'סעיד', 'כארם', 'אנס',
        # Additional common names
        'עבד', 'אבו', 'מוחמוד', 'מחמוד', 'סלים', 'פרח', 'פארח', 'פיסל',
        'פואד', 'פגיד', 'נסיר', 'אגס', 'חלב', 'שיח', "שיח'",
        # Names that appear as kunyas' complements (son names)
        'אחמד', 'סלאם', 'פאחר', 'עאדף', 'קדמה', 'פלס', 'בעאד', 'רהני',
        'חמסה', 'אלרהני',
    })

    # Cached normalized noise tokens (computed once)
    _noise_tokens_normalized: Optional[Set[str]] = field(default=None, repr=False)

    # Fix 29.38: Cached normalized common given names (computed once)
    # Used for consistent comparison with phonetically normalized tokens
    _common_given_names_normalized: Optional[Set[str]] = field(default=None, repr=False)

    def get_noise_tokens_normalized(self) -> Set[str]:
        """Get noise tokens with both Hebrew and ASCII punctuation variants."""
        if self._noise_tokens_normalized is None:
            normalized = set()
            for token in self.NOISE_TOKENS:
                normalized.add(token)
                # Add ASCII-punctuation variant
                ascii_variant = token.replace('״', '"').replace('׳', "'")
                normalized.add(ascii_variant)
            self._noise_tokens_normalized = normalized
        return self._noise_tokens_normalized

    def get_common_given_names_normalized(self) -> Set[str]:
        """Fix 29.38: Get common given names with phonetic normalization applied.

        COMMON_ARABIC_GIVEN_NAMES contains raw Hebrew names like 'עלי' (Ali with Ayin).
        When comparing with phonetically normalized tokens (where ע→א), we need the
        normalized versions like 'אלי' for proper membership testing.

        This ensures names like 'עלי' are correctly recognized as common given names
        even after phonetic normalization converts them to 'אלי'.
        """
        if self._common_given_names_normalized is None:
            # Import at method call time to avoid circular dependency
            # (normalize_arabic_phonetic is defined later in the file)
            self._common_given_names_normalized = {
                normalize_arabic_phonetic(name)
                for name in self.COMMON_ARABIC_GIVEN_NAMES
            }
        return self._common_given_names_normalized

    def validate(self) -> None:
        weight_sum = self.WEIGHT_CHAR_NGRAM + self.WEIGHT_TOKEN_SET + self.WEIGHT_IDF_JACCARD
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Similarity weights must sum to 1.0, got {weight_sum}")


# ============================================================================
# YANIS API CLIENT
# ============================================================================

class YanisAPIClient:
    """Yanis (יאניס) batch name-matching API client.

    Two pre-fetch calls per dataset:
    1. prefetch_veto_confirm(): edges 0.70-0.90, minScore=veto_threshold(60)
    2. prefetch_rescue(): candidates 0.45-0.69, minScore=rescue_threshold(85)

    After pre-fetching, score(n1, n2) is a pure cache lookup (no HTTP).
    """

    def __init__(self, url, token, timeout=30, max_batch_names=500):
        self.url = url
        self.token = token
        self.timeout = timeout
        self.max_batch_names = max_batch_names
        self._cache = {}              # {(min_name, max_name): score 0-100}
        self._queried_sets = {}       # {'veto': set_of_names, 'rescue': set_of_names}
        self._api_success = {}        # {'veto': bool, 'rescue': bool}
        self.stats = {
            'batch_calls': 0, 'pairs_returned': 0, 'errors': 0,
            'vetoed': 0, 'confirmed': 0, 'rescued': 0, 'discovered': 0,
            'attached': 0
        }
        self.decisions = []           # Structured log for phase gate review
        self.error_messages = []      # Capture exceptions for diagnostic display

    @staticmethod
    def _sym_key(n1, n2):
        return (min(n1, n2), max(n1, n2))

    def _batch_call(self, group1, group2, min_score):
        """POST to Yanis API. Returns {(n1,n2): score} on 0-100 scale."""
        payload = {
            "namesGroup1": group1,
            "namesGroup2": group2,
            "minScore": [int(min_score)],
            "limit": [999999],
            "punish_order": ["false"],
            "culture": ["Arabic"],
            "ordered_lists": False,
            "ignore_common_first_name": True
        }
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "Authorization": self.token
        }
        resp = requests.post(self.url, json=payload, headers=headers,
                             timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        results = {}
        for row in data:
            key = self._sym_key(row['name1'], row['name2'])
            # API returns 'score' as decimal (0-1); convert to 0-100 scale
            val = float(row['score']) * 100.0
            results[key] = val
            self._cache[key] = val
        self.stats['batch_calls'] += 1
        self.stats['pairs_returned'] += len(results)
        return results

    def prefetch_veto_confirm(self, candidate_names, min_score):
        """Call 1: Batch-fetch scores for veto/confirm edges (self-join)."""
        names = sorted(candidate_names)
        self._queried_sets['veto'] = set(names)
        try:
            self._batch_call(names, names, min_score)
            self._api_success['veto'] = True
        except Exception as e:
            logging.warning(f"[YANIS] Veto prefetch failed: {e}")
            self._api_success['veto'] = False
            self.stats['errors'] += 1
            self.error_messages.append(f"veto: {type(e).__name__}: {e}")

    def prefetch_rescue(self, candidate_names, min_score):
        """Call 2: Batch-fetch scores for rescue candidates (self-join)."""
        names = sorted(candidate_names)
        self._queried_sets['rescue'] = set(names)
        try:
            self._batch_call(names, names, min_score)
            self._api_success['rescue'] = True
        except Exception as e:
            logging.warning(f"[YANIS] Rescue prefetch failed: {e}")
            self._api_success['rescue'] = False
            self.stats['errors'] += 1
            self.error_messages.append(f"rescue: {type(e).__name__}: {e}")

    def prefetch_discovery(self, all_names, min_score):
        """Call 3: Batch-fetch ALL cluster names for discovery (self-join)."""
        names = sorted(all_names)
        if len(names) > self.max_batch_names:
            logging.warning(f"[YANIS] Discovery: {len(names)} names exceeds "
                           f"batch cap {self.max_batch_names}, truncating")
            names = names[:self.max_batch_names]
        self._queried_sets['discovery'] = set(names)
        try:
            self._batch_call(names, names, min_score)
            self._api_success['discovery'] = True
        except Exception as e:
            logging.warning(f"[YANIS] Discovery prefetch failed: {e}")
            self._api_success['discovery'] = False
            self.stats['errors'] += 1
            self.error_messages.append(f"discovery: {type(e).__name__}: {e}")

    def was_queried(self, batch_id, n1, n2):
        """Check if BOTH names were in the specified batch."""
        names = self._queried_sets.get(batch_id, set())
        return n1 in names and n2 in names

    def score(self, n1, n2):
        """Lookup pre-fetched score. Returns score (0-100) or None."""
        return self._cache.get(self._sym_key(n1, n2))

    def log_summary(self):
        return (f"[YANIS] batch_calls={self.stats['batch_calls']} "
                f"pairs={self.stats['pairs_returned']} errors={self.stats['errors']} "
                f"attached={self.stats['attached']} "
                f"vetoed={self.stats['vetoed']} confirmed={self.stats['confirmed']} "
                f"rescued={self.stats['rescued']} discovered={self.stats['discovered']}")

    # ------------------------------------------------------------------
    # Fix 30.1: Asymmetric prefetch for Stage 6.2 anchor attachment
    # ------------------------------------------------------------------
    def prefetch_anchor_attach(self, unresolved_names, anchor_names, min_score):
        """Asymmetric batch: namesGroup1 (unresolved) ≠ namesGroup2 (anchors).

        This is the key difference from veto/rescue/discovery which all do self-joins.
        Stage 6.2 needs unresolved cluster names compared against anchor names.
        """
        group1 = sorted(unresolved_names)
        group2 = sorted(anchor_names)
        if not group1 or not group2:
            self._api_success['anchor_attach'] = False
            return
        # Respect batch cap
        total = len(group1) + len(group2)
        if total > self.max_batch_names:
            logging.warning(f"[YANIS] Anchor attach: {total} names exceeds "
                           f"batch cap {self.max_batch_names}, truncating")
            # Prioritize anchor names (smaller set, always included)
            remaining = self.max_batch_names - len(group2)
            if remaining < 1:
                group2 = group2[:self.max_batch_names]
                group1 = group1[:1]
            else:
                group1 = group1[:remaining]
        self._queried_sets['anchor_attach'] = set(group1) | set(group2)
        try:
            self._batch_call(group1, group2, min_score)
            self._api_success['anchor_attach'] = True
        except Exception as e:
            logging.warning(f"[YANIS] Anchor attach prefetch failed: {e}")
            self._api_success['anchor_attach'] = False
            self.stats['errors'] += 1
            self.error_messages.append(f"anchor_attach: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Fix 30.2: Bundle-level score aggregation from cache
    # ------------------------------------------------------------------
    def best_scores_for_bundles(self, strong1, weak1, strong2, weak2):
        """Aggregate cached Yanis scores between two evidence bundles.

        Scans _cache (via _sym_key) for all pairs between the name lists.
        Returns dict with:
          best_strong_to_strong: highest score between strong1 x strong2
          support_count_ge90: count of strong1 x strong2 pairs scoring ≥ 90
          best_strong_to_weak: highest score between strong1 x weak2 or weak1 x strong2
          best_weak_to_weak: highest score between weak1 x weak2
        """
        best_s2s = 0.0
        support_ge90 = 0
        best_s2w = 0.0
        best_w2w = 0.0

        # strong vs strong
        for n1 in strong1:
            for n2 in strong2:
                sc = self._cache.get(self._sym_key(n1, n2))
                if sc is not None:
                    if sc > best_s2s:
                        best_s2s = sc
                    if sc >= 90.0:
                        support_ge90 += 1

        # strong vs weak (both directions)
        for n1 in strong1:
            for n2 in weak2:
                sc = self._cache.get(self._sym_key(n1, n2))
                if sc is not None and sc > best_s2w:
                    best_s2w = sc
        for n1 in weak1:
            for n2 in strong2:
                sc = self._cache.get(self._sym_key(n1, n2))
                if sc is not None and sc > best_s2w:
                    best_s2w = sc

        # weak vs weak
        for n1 in weak1:
            for n2 in weak2:
                sc = self._cache.get(self._sym_key(n1, n2))
                if sc is not None and sc > best_w2w:
                    best_w2w = sc

        return {
            'best_strong_to_strong': best_s2s,
            'support_count_ge90': support_ge90,
            'best_strong_to_weak': best_s2w,
            'best_weak_to_weak': best_w2w,
        }

    # ------------------------------------------------------------------
    # Fix 30.2: Batch prefetch for Stage 7 bundle scoring
    # ------------------------------------------------------------------
    def prefetch_stage7_bundles(self, bundle_names_by_cid, min_score):
        """Collect ALL bundle names for Stage 7 Phases 1-3 in one batch call.

        Args:
            bundle_names_by_cid: Dict[cluster_id, Set[str]] of API names
            min_score: minimum score for the API call
        """
        all_names = set()
        for names in bundle_names_by_cid.values():
            all_names |= names
        names = sorted(all_names)
        if not names:
            self._api_success['stage7_bundles'] = False
            return
        if len(names) > self.max_batch_names:
            logging.warning(f"[YANIS] Stage7 bundles: {len(names)} names exceeds "
                           f"batch cap {self.max_batch_names}, truncating")
            names = names[:self.max_batch_names]
        self._queried_sets['stage7_bundles'] = set(names)
        try:
            self._batch_call(names, names, min_score)
            self._api_success['stage7_bundles'] = True
        except Exception as e:
            logging.warning(f"[YANIS] Stage7 bundles prefetch failed: {e}")
            self._api_success['stage7_bundles'] = False
            self.stats['errors'] += 1
            self.error_messages.append(f"stage7_bundles: {type(e).__name__}: {e}")


def _api_representative_name(sig, allow_single_token=False):
    """Pick the best name to send to Yanis (≥2 tokens preferred)."""
    # Prefer API-ready (lightly normalized) names
    api = sig.api_name
    if api and len(api.split()) >= 2:
        return api
    api_names = sig.all_names_api
    candidates = [n for n in api_names if len(n.split()) >= 2]
    if candidates:
        return max(candidates, key=len)
    # Fallback to fully-normalized names
    canonical = sig.canonical_name
    if canonical and len(canonical.split()) >= 2:
        return canonical
    candidates = [n for n in sig.all_names_normalized if len(n.split()) >= 2]
    if candidates:
        return max(candidates, key=len)
    if allow_single_token:
        for pool in [api, canonical]:
            if pool and pool.strip():
                return pool
        for pool in [api_names, sig.all_names_normalized]:
            single = [n for n in pool if n and n.strip()]
            if single:
                return max(single, key=len)
    return None


# ============================================================================
# ENTITY ID HELPER
# ============================================================================

def _coerce_entity_id(val) -> Optional[str]:
    """Safely convert entity ID to string, handling nulls, floats, and edge cases.

    - Returns None for null/empty/whitespace/"nan"/"none"
    - Converts whole-number floats to ints before stringifying (123.0 → "123")
    - Handles 0 correctly (doesn't treat as falsy)
    """
    if val is None:
        return None
    if isinstance(val, float):
        if val != val:  # NaN check
            return None
        if val == int(val):  # Whole number float
            val = int(val)
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass  # Not a pandas-compatible type

    s = str(val).strip()
    if not s or s.lower() in ('nan', 'none', 'null', ''):
        return None

    return s


def _split_nicknames(nicknames_str: Optional[str]) -> List[str]:
    """Split nicknames string by comma, Arabic comma, semicolon, or brackets.

    Handles cube1 verified_nicknames and cube2 contact nicknames consistently.
    Returns list of stripped, non-empty nicknames.

    Fix 17.1: Added parentheses/brackets to split pattern.
    Previously "Jojo (The King)" became one nickname "Jojo The King" after normalization,
    causing "Jojo" to score ~0.73 (below 0.75 threshold) due to fuzz.ratio averaging.
    Now it splits into ["Jojo", "The King"], so "Jojo" matches with score 1.0.
    """
    if not nicknames_str:
        return []
    import re as regex
    result = []
    # Fix 17.1: Added () [] {} to split pattern for compound nicknames
    for nn in regex.split(r'[,،;|/\\\n()\[\]{}]+', str(nicknames_str)):
        nn_clean = nn.strip()
        if nn_clean:
            result.append(nn_clean)
    return result


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class NameMention:
    """A single mention of a name in the call data."""
    mention_id: str
    raw_text: str
    normalized: str
    tokens: List[str]
    phone: str
    call_id: str
    date: Optional[str] = None
    side: str = ''
    other_phone: str = ''
    segment_index: int = 0
    segment_count: int = 1
    original_field: str = ''
    verified_name: Optional[str] = None
    verified_nicknames: Optional[str] = None
    verified_entity_id: Optional[str] = None  # Segment-level: only on best-matching segment
    row_verified_entity_id: Optional[str] = None  # Row-level: always present if row had it (for provenance)
    entity_id_assignment: str = ''  # 'confident' | 'fallback' | 'ambiguous' | ''
    verified_status: Optional[str] = None  # Row-level: entity status from cube3 via Point
    verified_id_number: Optional[str] = None  # Row-level: entity id_number from cube3 via Point
    is_blmz: bool = False
    must_not_link: Set[str] = field(default_factory=set)
    cluster_id: Optional[str] = None


@dataclass
class Cube2Match:
    """Result of matching a mention/cluster against cube2 contacts."""
    name: str
    nickname: Optional[str]
    score: float
    second_score: float
    margin: float
    entity_id: Optional[str] = None

    # v9 additions (for cube2-first-class usage)
    # contact_key is stable within a phone and is used for intra-phone bridging.
    contact_key: Optional[str] = None  # 'EID:<id>' or 'NAME:<hash>'
    name_normalized: str = ''
    tokens: List[str] = field(default_factory=list)
    quality_tier: str = ''  # LOW / MED / HIGH
    quality_score: float = 0.0
    # Fix 44: Source phone for cross-phone contact lookups
    # When match_global() finds a contact on Phone B but cluster is on Phone A,
    # we need to know the source phone to correctly retrieve contact details.
    source_phone: Optional[str] = None
    status: Optional[str] = None  # Entity status from cube2
    id_number: Optional[str] = None  # Entity id_number from cube2

    @property
    def is_confident(self) -> bool:
        # Backwards-compatible default; resolver may apply stricter rules.
        return self.margin >= 0.15


@dataclass
class EntityCluster:
    """A resolved entity cluster containing related mentions."""
    cluster_id: str
    phone: str
    mentions: List[NameMention] = field(default_factory=list)
    mention_ids: Set[str] = field(default_factory=set)
    canonical_name: str = ''
    resolution_type: str = ''
    confidence: str = ''
    nicknames: Set[str] = field(default_factory=set)
    best_score: float = 0.0
    second_best_score: float = 0.0
    score_margin: float = 0.0
    match_evidence: str = ''
    cube2_candidates: str = ''
    flags: List[str] = field(default_factory=list)
    cross_phone_links: List[Dict[str, str]] = field(default_factory=list)
    global_entity_id: str = ''  # Fix 2.2: Cross-phone unified identity
    verified_entity_id: Optional[str] = None  # Entity ID from cube1 or cube2
    verified_status: Optional[str] = None  # Entity status from cube1/cube2
    verified_id_number: Optional[str] = None  # Entity id_number from cube1/cube2
    merge_reason: str = ''  # v8: Explains WHY this cluster joined its global entity
    # v9: cube2/phonebook metadata (internal; does not change output schema)
    cube2_contact_key: Optional[str] = None
    phonebook_quality: str = ''  # LOW / MED / HIGH
    phonebook_quality_score: float = 0.0
    display_name: str = ''  # Raw/original text for output; canonical_name stays for matching

    @property
    def size(self) -> int:
        return len(self.mentions)

    def has_verified(self) -> bool:
        return any(m.verified_name for m in self.mentions)


# ============================================================================
# HELPER UTILITIES
# ============================================================================

# Hebrew Unicode ranges
HEBREW_LETTERS_RANGE = (0x05D0, 0x05EA)
HEBREW_NIQQUD_RANGE = (0x0591, 0x05C7)


def remove_niqqud(text: str) -> str:
    """Remove Hebrew niqqud (vowel marks) from text."""
    if not text:
        return ''
    result = []
    for char in text:
        code = ord(char)
        if not (HEBREW_NIQQUD_RANGE[0] <= code <= HEBREW_NIQQUD_RANGE[1]):
            result.append(char)
    return ''.join(result)


def normalize_hebrew_punctuation(text: str) -> str:
    """Normalize Hebrew-specific punctuation marks."""
    if not text:
        return ''
    text = text.replace('־', '-')  # Hebrew maqaf to hyphen
    text = text.replace('״', '"')  # Hebrew gershayim
    text = text.replace('׳', "'")  # Hebrew geresh
    return text


def normalize_kunya_spacing(text: str) -> str:
    """Normalize spacing around kunya prefix (only אבו) and אל prefix."""
    if not text:
        return ''
    # Only אבו is treated as kunya in this algorithm
    text = re.sub(r'אבו[\s\-]+', 'אבו-', text)
    # אל is the definite article, not kunya
    text = re.sub(r'אל[\s\-]+', 'אל-', text)
    return text


def normalize_kunya_alias_token(token: str) -> str:
    """Canonicalize kunya article variants.

    Example: "אבו-אלרהני" and "אבו-רהני" should be treated as equivalent.
    """
    if not token:
        return ''
    tok = str(token).strip()
    if not tok:
        return ''
    tok = re.sub(r'^אבו[\s\-]+אל[\s\-]*', 'אבו-', tok)
    tok = re.sub(r'^אבו[\s]+', 'אבו-', tok)
    tok = re.sub(r'-{2,}', '-', tok)
    return tok


def normalize_kunya_alias_text(text: str) -> str:
    """Apply kunya alias normalization token-wise across a full string."""
    if not text:
        return ''
    return ' '.join(normalize_kunya_alias_token(tok) for tok in str(text).split())


def normalize_arabic_phonetic(text: str) -> str:
    """Normalize Arabic-Hebrew transliteration homophones for comparison.

    Fix 10.2: Arabic names transliterated to Hebrew often use different letters
    for the same sound, causing false conflicts in similarity matching:
    - ק vs כ (Qof/Kaf) - both used for Arabic qāf/kāf sounds
    - ט vs ת (Tet/Tav) - both used for Arabic ṭāʾ/tāʾ sounds
    - ע vs א (Ayin/Alef) - both used for Arabic 'ayn/hamza in informal text

    Example: קאסם vs כאסם (Qassem) are the SAME person, not a conflict.
    Without normalization: 75% similarity → flagged as conflict.
    With normalization: 100% similarity → correctly identified as same person.

    Fix 24.1: Handle Geresh-modified letters used in Arabic transliteration.
    Hebrew uses geresh (׳) to represent Arabic sounds that don't exist in Hebrew:
    - ג׳ (Jim) = Arabic ج - maps to ג (Gimel)
    - ח׳ (Kha) = Arabic خ - maps to ח (Het)
    - כ׳ (Kha) = Arabic خ - maps to ח (Het, unified with ח׳ to prevent שיכ׳→שיך)
    - צ׳ (Cha) = Arabic تش - maps to צ (Tsade)
    - ר׳ (Ghayin) = Arabic غ - maps to ר (Resh)
    Without this fix: אחג׳ד vs אחגד are treated as different names.

    Fix 26.2: Manzapach Auto-Correction - Fix invalid Hebrew grammar where
    regular letters appear at word endings instead of their final (Sofit) forms.
    Hebrew Rule: כ, מ, נ, פ, צ CANNOT end a word - must be ך, ם, ן, ף, ץ.
    Many business systems produce invalid data like "ע׳חאנ" instead of "ע׳חאן".

    Fix 26.7: Doubled Letter Normalization - In Arabic transliteration to Hebrew,
    doubled letters represent long vowels (וו, יי, אא) or gemination (ממ, ננ).
    Example: "אלגמשראווי" vs "אלגמשראוי", "מוחממד" vs "מוחמד" are the SAME name.
    Without this fix: index lookup fails for spelling variants → no phonebook match.
    """
    if not text:
        return ''

    # Fix 24.1: Strip Geresh-modified letters to their base Hebrew form
    # Must be done BEFORE the standard phonetic mappings
    # Handle both Hebrew geresh (׳) and ASCII apostrophe (') variants
    text = text.replace("ג׳", "ג").replace("ג'", "ג")  # Jim → Gimel (ج)
    text = text.replace("ח׳", "ח").replace("ח'", "ח")  # Kha → Het (خ)
    # Fix: כ׳ also represents Arabic خ (Kha), same sound as ח׳. Map to ח (not כ)
    # to unify both variants AND prevent manzapach from turning word-final כ→ך.
    # Example: "שיכ׳" (Sheikh) → "שיח" (matches "שיח׳" correctly), not "שיך".
    text = text.replace("כ׳", "ח").replace("כ'", "ח")  # Kha → Het (خ) — unified with ח׳
    text = text.replace("צ׳", "צ").replace("צ'", "צ")  # Cha → Tsade (تش)
    text = text.replace("ר׳", "ר").replace("ר'", "ר")  # Ghayin → Resh (غ)
    text = text.replace("ז׳", "ז").replace("ז'", "ז")  # Zhe → Zayin (ژ)
    text = text.replace("ת׳", "ת").replace("ת'", "ת")  # Tha → Tav (ث)
    text = text.replace("ד׳", "ד").replace("ד'", "ד")  # Dhal → Dalet (ذ)
    # Fix 29.16: Add missing Ayin with geresh rule
    # ע׳ is used for Arabic 'ayn (ع) sound in some transliteration systems.
    # Without this rule, ע׳אחן → עאחן → אאחן (double alef after ע→א mapping)
    # With this rule, ע׳אחן → עאחן → אאחן is avoided because geresh is stripped first.
    # Must strip BEFORE the ע → א mapping at line 455.
    text = text.replace("ע׳", "ע").replace("ע'", "ע")  # Ayin with geresh → Ayin (ع)

    # Clean up any remaining standalone geresh/apostrophe marks
    text = text.replace("׳", "").replace("'", "")

    # Map to canonical form (using the more common letter)
    # Must be done BEFORE Manzapach fix so that ק→כ at word end becomes ך
    text = text.replace('ק', 'כ')  # Qof → Kaf
    text = text.replace('ט', 'ת')  # Tet → Tav
    text = text.replace('ע', 'א')  # Ayin → Alef

    # Fix 26.7: Doubled Letter Normalization
    # In Arabic transliteration to Hebrew, doubled letters represent:
    # - Long vowels: וו (ū), יי (ī), אא (ā)
    # - Gemination: ממ, ננ, ססּ (emphatic consonants in Arabic)
    # The doubling is stylistic/phonetic, not a meaningful identity distinction.
    # Examples: "אלגמשראווי" = "אלגמשראוי", "מוחממד" = "מוחמד"
    # Use regex to normalize ANY doubled Hebrew letter to single.
    text = re.sub(r'([\u0590-\u05FF])\1+', r'\1', text)

    # Fix 26.2: Manzapach Auto-Correction
    # Fix invalid Hebrew where regular letters appear at word endings
    # Must be done AFTER phonetic mapping so ק→כ at end becomes ך
    # Example: "יצחק" → phonetic → "יצחכ" → manzapach → "יצחך"
    manzapach_to_sofit = {'כ': 'ך', 'מ': 'ם', 'נ': 'ן', 'פ': 'ף', 'צ': 'ץ'}
    tokens = text.split()
    fixed_tokens = []
    for token in tokens:
        if token and token[-1] in manzapach_to_sofit:
            token = token[:-1] + manzapach_to_sofit[token[-1]]
        fixed_tokens.append(token)
    text = ' '.join(fixed_tokens)

    return text


# ── Fix 31.1: Confusable Given Name Safety Gate ──────────────────────────

def _detect_confusable_given_names(
    name1: str, name2: str,
    common_given_names_normalized: Set[str],
    min_sim: int = 74, max_identity: int = 97,
    family_match_min: int = 80,
) -> Tuple[bool, str]:
    """Detect if two names have confusable but DIFFERENT Arabic given names.

    Returns (is_confusable, reason_string).

    "Confusable" means the given names are character-similar (e.g. מחמד vs מחמוד,
    ratio ~89%) but NOT identical.  Two genuinely identical given names (ratio ≥ 97)
    or two names with matching family names are NOT confusable.
    """
    if not name1 or not name2:
        return False, ''

    toks1 = name1.split()
    toks2 = name2.split()
    if not toks1 or not toks2:
        return False, ''

    # ── Extract given name (first non-kunya token) ──
    def _extract_given_family(toks):
        non_kunya = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if t == 'אבו' and i + 1 < len(toks):
                if toks[i + 1] == 'אל' and i + 2 < len(toks):
                    i += 3  # "אבו אל X" — skip article too
                else:
                    i += 2  # "אבו X" — normal complement
                continue
            if t.startswith('אבו-'):
                i += 1
                continue
            non_kunya.append(t)
            i += 1
        if not non_kunya:
            return None, None
        given = normalize_arabic_phonetic(non_kunya[0])
        family = normalize_arabic_phonetic(non_kunya[-1]) if len(non_kunya) >= 2 else None
        return given, family

    given1, family1 = _extract_given_family(toks1)
    given2, family2 = _extract_given_family(toks2)

    if not given1 or not given2:
        return False, ''

    # Both given names must be common Arabic names
    if given1 not in common_given_names_normalized:
        return False, ''
    if given2 not in common_given_names_normalized:
        return False, ''

    # Compute similarity between the two given names
    given_ratio = _char_ratio(given1, given2)

    # Not confusable if too dissimilar (below floor) or identical (above ceiling)
    if given_ratio < min_sim or given_ratio >= max_identity:
        return False, ''

    # Family override: if both have families and families match, NOT confusable
    if family1 and family2:
        family_ratio = _char_ratio(family1, family2)
        if family_ratio >= family_match_min:
            return False, ''

    reason = f"given1={given1},given2={given2},ratio={given_ratio:.1f}"
    return True, reason


def _names_have_confusable_given(
    names1: list, names2: list,
    common_given_names_normalized: Set[str],
    min_sim: int = 74, max_identity: int = 97,
    family_match_min: int = 80,
) -> Tuple[bool, str]:
    """Check if ANY pair from two name lists has confusable given names.

    names1, names2 are lists of name strings (e.g. bundle strong_names_api).
    Returns (is_confusable, first_reason_string).
    """
    for n1 in names1:
        if not n1 or not n1.strip():
            continue
        for n2 in names2:
            if not n2 or not n2.strip():
                continue
            confusable, reason = _detect_confusable_given_names(
                n1, n2, common_given_names_normalized,
                min_sim, max_identity, family_match_min)
            if confusable:
                return True, reason
    return False, ''


def normalize_phone(phone: Optional[str]) -> str:
    """Normalize phone number to consistent format."""
    if not phone or not isinstance(phone, str):
        return ''
    # Handle float→string artifact: "972501234567.0" → "972501234567"
    # DataFrame phone columns with NaN values become float64, and str() adds ".0"
    if phone.endswith('.0'):
        phone = phone[:-2]
    digits = re.sub(r'\D', '', phone)
    if not digits:
        return ''
    if digits.startswith('972') or digits.startswith('970'):
        digits = '0' + digits[3:]
    if len(digits) == 9 and not digits.startswith('0'):
        digits = '0' + digits
    return digits


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for JSON serialization."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        elif df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(0)
        elif df[col].dtype == 'bool':
            df[col] = df[col].fillna(False)
    return df


def create_error_df(error_type: str, message: str) -> pd.DataFrame:
    """Create standardized error DataFrame."""
    return pd.DataFrame({
        '_error': [True],
        '_error_type': [error_type],
        '_error_message': [message]
    })


# ============================================================================
# NORMALIZATION
# ============================================================================

class Normalizer:
    """Light text normalization - format level only."""

    def __init__(self, config: Config):
        self.config = config
        self._whitespace_pattern = re.compile(r'\s+')
        # Include parentheses/brackets to prevent "(Driver)" from hijacking identity anchors
        self._separator_pattern = re.compile(r'[,/\\|()\[\]{}]')
        self._percentage_pattern = re.compile(r'\d+%')
        self._number_pattern = re.compile(r'^\d+$')

    def normalize(self, text: str) -> str:
        """Normalize text without losing semantic meaning."""
        if not text or not isinstance(text, str):
            return ''
        text = unicodedata.normalize('NFKC', text)
        text = remove_niqqud(text)
        text = normalize_hebrew_punctuation(text)
        text = normalize_kunya_spacing(text)
        text = normalize_kunya_alias_text(text)

        # Fix 7.1: Normalize repeated quotes and common abbreviation variants
        # Example: 'ב׳׳ר' (double geresh) → 'בר' so it can be treated as a noise/connector token.
        text = re.sub(r"'{2,}", "'", text)
        text = re.sub(r'"{2,}', '"', text)
        text = re.sub(r'\bב[\'"]+ר\b', 'בר', text)

        # Fix 5.4a: Decompose concatenated אל forms
        # Pattern: consonant + אל + consonant → consonant + space + אל + space + consonant
        # Examples: 'עבדאלרחמן' → 'עבד אל רחמן', 'אלקאדר' → 'אל קאדר'
        text = re.sub(
            r'([אבגדהוזחטיכלמנסעפצקרשת])אל([אבגדהוזחטיכלמנסעפצקרשת])',
            r'\1 אל \2',
            text
        )

        # Fix 5.4b: Convert dash variants to spaces ONLY when surrounded by spaces.
        # This preserves kunya-style hyphens (אבו-מחמד) set by normalize_kunya_spacing,
        # while still splitting "name - other" patterns.
        # Includes: ASCII hyphen (-), Hebrew maqaf (־), en-dash (–), em-dash (—)
        text = re.sub(r'\s+[-־–—]\s+', ' ', text)

        text = self._separator_pattern.sub(' ', text)
        text = self._whitespace_pattern.sub(' ', text)
        text = text.strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Extract tokens from normalized text.

        Note: Hyphen handling is done in normalize() - all dash variants are
        converted to spaces there, so tokenize() just does split + filter.
        """
        if not text:
            return []

        tokens = text.split()

        # Filter out short tokens, numbers, percentages
        filtered = []
        for token in tokens:
            if len(token) < 2:
                continue
            if self._number_pattern.match(token):
                continue
            if self._percentage_pattern.match(token):
                continue
            filtered.append(token)
        return filtered


_API_RE_SINGLE_QUOTES = re.compile(r"'{2,}")
_API_RE_DOUBLE_QUOTES = re.compile(r'"{2,}')
_API_RE_BAR = re.compile(r'\bב[\'"]+ר\b')
_API_RE_AL_DECOMP = re.compile(
    r'([אבגדהוזחטיכלמנסעפצקרשת])אל([אבגדהוזחטיכלמנסעפצקרשת])'
)
_API_RE_DASH_SPACE = re.compile(r'\s+[-\u05be\u2013\u2014]\s+')
_API_RE_SEPARATORS = re.compile(r'[,/\\|()\[\]{}]')
_API_RE_WHITESPACE = re.compile(r'\s+')


def normalize_for_api(text: str) -> str:
    """Light normalization for Yanis API — preserves name structure.

    Keeps al-decomposition (compound names need ≥2 tokens for API).
    Omits kunya spacing (would join tokens → fail ≥2 token check).
    Omits kunya alias (would remove אל content from names).
    """
    if not text or not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = remove_niqqud(text)
    text = normalize_hebrew_punctuation(text)
    # Collapse repeated quotes (same as normalize Fix 7.1)
    text = _API_RE_SINGLE_QUOTES.sub("'", text)
    text = _API_RE_DOUBLE_QUOTES.sub('"', text)
    text = _API_RE_BAR.sub('בר', text)
    # Al-decomposition — KEEP: compound names need ≥2 tokens for API gate
    text = _API_RE_AL_DECOMP.sub(r'\1 אל \2', text)
    # Dash-to-space only when surrounded by spaces (Fix 5.4b)
    text = _API_RE_DASH_SPACE.sub(' ', text)
    # Remove separators, collapse whitespace
    text = _API_RE_SEPARATORS.sub(' ', text)
    text = _API_RE_WHITESPACE.sub(' ', text).strip()
    return text


# ============================================================================
# MENTION EXPLOSION
# ============================================================================

# Filler tokens that don't constitute valid name content
FILLER_TOKENS: Set[str] = {'בנוסף', 'כן', 'אולי', 'וכו', 'לפי', 'עם', 'את', 'של', 'על', 'או', 'גם'}

# Allowed singleton tokens (like בלמ״ז) that are valid even as single tokens
ALLOWED_SINGLETONS: Set[str] = {'בלמ״ז', 'בלמז', 'דובר לא מזוהה'}


def validate_mention(normalized: str, tokens: List[str]) -> Tuple[bool, str]:
    """Validate if a mention should create an entity."""
    if not normalized or not tokens:
        return (False, 'EMPTY_AFTER_STRIP')
    normalized_lower = normalized.lower()
    for allowed in ALLOWED_SINGLETONS:
        if allowed in normalized_lower:
            return (True, 'BLMZ')
    if len(tokens) == 1:
        if tokens[0] in FILLER_TOKENS:
            return (False, 'FILLER_ONLY')
    non_filler = [t for t in tokens if t not in FILLER_TOKENS]
    if not non_filler:
        return (False, 'ALL_FILLER')
    return (True, 'VALID')


class MentionExploder:
    """Explode call rows into individual mention segments."""

    def __init__(self, config: Config):
        self.config = config
        self.normalizer = Normalizer(config)
        self._plus_pattern = re.compile(r'\s*\+\s*')

    def explode_dataframe(self, df: pd.DataFrame) -> List[NameMention]:
        """Process entire DataFrame into list of mentions."""
        all_mentions = []
        
        # DIAGNOSTIC: Check if verified_entity_id columns exist and have values
        eid_col_a = self.config.COL_VERIFIED_ENTITY_ID_A
        eid_col_b = self.config.COL_VERIFIED_ENTITY_ID_B
        
        has_eid_col_a = eid_col_a in df.columns
        has_eid_col_b = eid_col_b in df.columns
        
        print(f"[DIAG] verified_entity_id columns present: A={has_eid_col_a}, B={has_eid_col_b}")
        print(f"[DIAG] All input columns: {list(df.columns)}")
        
        if has_eid_col_a:
            non_empty_a = df[eid_col_a].apply(lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None')).sum()
            print(f"[DIAG] verified_entity_id_A: {non_empty_a}/{len(df)} rows have non-empty values")
            if non_empty_a > 0:
                sample = df[df[eid_col_a].apply(lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None'))][eid_col_a].head(3).tolist()
                print(f"[DIAG] Sample verified_entity_id_A values: {sample}")
        
        if has_eid_col_b:
            non_empty_b = df[eid_col_b].apply(lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None')).sum()
            print(f"[DIAG] verified_entity_id_B: {non_empty_b}/{len(df)} rows have non-empty values")
            if non_empty_b > 0:
                sample = df[df[eid_col_b].apply(lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None'))][eid_col_b].head(3).tolist()
                print(f"[DIAG] Sample verified_entity_id_B values: {sample}")
        
        for idx, row in df.iterrows():
            mentions_a = self._explode_row(row, 'A')
            all_mentions.extend(mentions_a)
            mentions_b = self._explode_row(row, 'B')
            all_mentions.extend(mentions_b)
        
        # DIAGNOSTIC: Count mentions with verified_entity_id
        mentions_with_eid = sum(1 for m in all_mentions if m.verified_entity_id)
        print(f"[DIAG] Mentions with verified_entity_id: {mentions_with_eid}/{len(all_mentions)}")
        
        return all_mentions

    def _explode_row(self, row: pd.Series, side: str) -> List[NameMention]:
        """Process one side of a call row into mentions."""
        clean_name = self._get_clean_name(row, side)
        verified_name = self._get_verified_name(row, side)
        verified_nicknames = self._get_verified_nicknames(row, side)
        verified_entity_id = self._get_verified_entity_id(row, side)
        verified_status = self._get_verified_status(row, side)
        verified_id_number = self._get_verified_id_number(row, side)
        phone = self._get_phone(row, side)
        other_phone = self._get_other_phone(row, side)
        call_id = str(row.get(self.config.COL_CALL_ID, ''))
        date = str(row.get(self.config.COL_DATE, '')) if self.config.COL_DATE in row else None

        if not clean_name or pd.isna(clean_name):
            return []

        segments = self._split_segments(clean_name)
        verified_segment_idx = self._find_best_verified_match(segments, verified_name, verified_nicknames)

        # PHASE 1 FIX: Track which segments pass validation BEFORE assigning entity_id
        # This allows fallback assignment when best segment is invalid.
        #
        # BLMZ SAFETY: Treat "unknown speaker" markers as a *negative constraint* for identity IDs.
        # They may carry a row-level verified_entity_id (line ownership), but we must NOT assign
        # that ID to the BLMZ segment (segment-level), otherwise the mention can hard-link into a
        # real person's cluster and corrupt resolved_name.
        valid_segment_indices = []
        valid_non_blmz_segment_indices = []
        segment_data = []  # (idx, segment, normalized, tokens, is_valid, validity_reason, is_blmz)

        for idx, segment in enumerate(segments):
            normalized = self.normalizer.normalize(segment)
            tokens = self.normalizer.tokenize(normalized)
            is_valid, validity_reason = validate_mention(normalized, tokens)
            is_blmz = self._is_blmz(normalized)
            segment_data.append((idx, segment, normalized, tokens, is_valid, validity_reason, is_blmz))
            if is_valid:
                valid_segment_indices.append(idx)
                if not is_blmz:
                    valid_non_blmz_segment_indices.append(idx)

        # Determine entity_id assignment with fallback
        entity_id_segment_idx = None
        entity_id_assignment = ''

        if verified_entity_id:
            if verified_segment_idx is not None:
                # Best match found (but never assign IDs onto BLMZ segments)
                if verified_segment_idx in valid_non_blmz_segment_indices:
                    entity_id_segment_idx = verified_segment_idx
                    entity_id_assignment = 'confident'
                elif valid_non_blmz_segment_indices:
                    # Best segment failed validation OR was BLMZ - fallback to first non-BLMZ valid segment
                    entity_id_segment_idx = valid_non_blmz_segment_indices[0]
                    entity_id_assignment = 'fallback'
                else:
                    # No non-BLMZ valid segments exist (all valid segments are BLMZ markers)
                    entity_id_assignment = 'ambiguous'
            elif valid_non_blmz_segment_indices:
                # No verified_name match - check if verified_name was provided
                if verified_name:
                    # CRITICAL: verified_name exists but doesn't match ANY segment.
                    # This is a data mismatch (e.g., "Dani + Yossi" with verified="Moshe").
                    # Do NOT assign Moshe's ID to Dani - this would corrupt identity links.
                    # It's safer to lose the ID than to pollute the graph with false links.
                    entity_id_assignment = 'mismatch_no_assign'
                else:
                    # No verified_name at all - ID might reasonably apply to first speaker
                    entity_id_segment_idx = valid_non_blmz_segment_indices[0]
                    entity_id_assignment = 'fallback_no_verified'
            else:
                entity_id_assignment = 'ambiguous'

        mentions = []
        mention_ids = []

        for idx, segment, normalized, tokens, is_valid, validity_reason, is_blmz in segment_data:
            if not is_valid:
                continue

            mention_id = f"{call_id}_{side}_{idx}"
            mention_ids.append(mention_id)

            mention = NameMention(
                mention_id=mention_id,
                raw_text=segment.strip(),
                normalized=normalized,
                tokens=tokens,
                phone=phone,
                call_id=call_id,
                date=date,
                side=side,
                other_phone=other_phone,
                segment_index=idx,
                segment_count=len(segments),
                original_field=clean_name,
                verified_name=verified_name if idx == verified_segment_idx else None,
                verified_nicknames=verified_nicknames if idx == verified_segment_idx else None,
                # Segment-level: only on best-matching valid segment
                verified_entity_id=verified_entity_id if (idx == entity_id_segment_idx and not is_blmz) else None,
                # Row-level: ALWAYS present if row had it (for provenance/backbone)
                row_verified_entity_id=verified_entity_id,
                entity_id_assignment=entity_id_assignment if idx == entity_id_segment_idx else '',
                is_blmz=is_blmz,
                verified_status=verified_status if (idx == entity_id_segment_idx and not is_blmz) else None,
                verified_id_number=verified_id_number if (idx == entity_id_segment_idx and not is_blmz) else None,
            )
            mentions.append(mention)

        # Build must_not_link, but skip pairs that are likely aliases (Fix 1.4)
        if len(mentions) > 1:
            for i, m in enumerate(mentions):
                must_not_link_ids = set()
                for j, other_m in enumerate(mentions):
                    if i == j:
                        continue
                    # Only add to must_not_link if NOT an alias
                    if not self._segments_are_aliases(m.tokens, other_m.tokens):
                        must_not_link_ids.add(other_m.mention_id)
                m.must_not_link = must_not_link_ids

        return mentions

    def _split_segments(self, text: str) -> List[str]:
        """Split text by '+' delimiter."""
        segments = self._plus_pattern.split(text)
        return [s.strip() for s in segments if s.strip()]

    def _find_best_verified_match(
        self,
        segments: List[str],
        verified_name: Optional[str],
        verified_nicknames: Optional[str]
    ) -> Optional[int]:
        """Find segment that best matches verified identity."""
        if not verified_name:
            return None
        if len(segments) == 1:
            return 0

        targets = [verified_name]
        if verified_nicknames:
            targets.extend(_split_nicknames(verified_nicknames))

        best_idx = None
        best_score = 0.0

        for idx, segment in enumerate(segments):
            seg_norm = self.normalizer.normalize(segment)
            seg_tokens = set(self.normalizer.tokenize(seg_norm))

            for target in targets:
                target_norm = self.normalizer.normalize(target)
                target_tokens = set(self.normalizer.tokenize(target_norm))

                if not seg_tokens or not target_tokens:
                    continue

                intersection = seg_tokens & target_tokens
                min_size = min(len(seg_tokens), len(target_tokens))

                # Fix: Al-prefix fallback for Arabic names in "+" splits.
                # "אלואדיה" (segment) should match "ואדיה" (verified) — אל is
                # the Arabic definite article ال.  Only activated when exact
                # token intersection is insufficient, to avoid any side-effects
                # on names that already match exactly.
                if len(intersection) < min_size:
                    def _al_strip(t):
                        return t[2:] if t.startswith('אל') and len(t) > 2 else t
                    seg_al = {_al_strip(t) for t in seg_tokens}
                    tgt_al = {_al_strip(t) for t in target_tokens}
                    inter_al = seg_al & tgt_al
                    if len(inter_al) > len(intersection):
                        intersection = inter_al
                        min_size = min(len(seg_al), len(tgt_al))

                # Fix 9.1: Prevent first-name-only matches when both sides have 2+ tokens.
                # "Yossi Levi" vs "Yossi Cohen" → intersection=1, min_size=2 → BLOCKED
                # This prevents assigning Cohen's entity_id to a Levi mention.
                if len(intersection) == 1 and min_size > 1:
                    continue  # Skip this target, insufficient overlap

                score = len(intersection) / min_size if min_size > 0 else 0.0

                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_score >= self.config.VERIFIED_MATCH_THRESHOLD:
            return best_idx
        return None

    def _is_blmz(self, normalized: str) -> bool:
        """Check if mention is an 'unknown speaker' marker."""
        check_text = normalized.lower()
        for blmz_token in self.config.BLMZ_TOKENS:
            blmz_normalized = self.normalizer.normalize(blmz_token).lower()
            if blmz_normalized in check_text:
                return True
        return False

    def _segments_are_aliases(self, tokens1: List[str], tokens2: List[str]) -> bool:
        """Check if two segments are likely aliases (same person, expanded form) (Fix 1.4).

        Safe rules:
        - ≥2 shared meaningful tokens
        - OR full containment with ≥2 tokens in shorter
        - OR 1-token prefix expansion (first name match)
        """
        noise = self.config.get_noise_tokens_normalized()
        set1 = set(tokens1) - noise
        set2 = set(tokens2) - noise
        list1 = [t for t in tokens1 if t not in noise]
        list2 = [t for t in tokens2 if t not in noise]

        if not set1 or not set2:
            return False

        shared = set1 & set2

        # Rule 1: ≥2 shared meaningful tokens
        if len(shared) >= 2:
            return True

        # Rule 2: Full containment with ≥2 tokens in shorter
        shorter_set, longer_set = (set1, set2) if len(set1) <= len(set2) else (set2, set1)
        if shorter_set.issubset(longer_set) and len(shorter_set) >= 2:
            return True

        # Rule 3: 1-token prefix expansion (משה ↔ משה אלקנה)
        shorter_list, longer_list = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
        if len(shorter_list) == 1 and len(longer_list) >= 2:
            if shorter_list[0] == longer_list[0]:  # First name match
                return True

        return False

    def _get_clean_name(self, row: pd.Series, side: str) -> Optional[str]:
        col = self.config.COL_CLEAN_NAME_A if side == 'A' else self.config.COL_CLEAN_NAME_B
        val = row.get(col)
        return str(val) if val and not pd.isna(val) else None

    def _get_verified_name(self, row: pd.Series, side: str) -> Optional[str]:
        col = self.config.COL_VERIFIED_NAME_A if side == 'A' else self.config.COL_VERIFIED_NAME_B
        val = row.get(col)
        return str(val) if val and not pd.isna(val) else None

    def _get_verified_nicknames(self, row: pd.Series, side: str) -> Optional[str]:
        col = self.config.COL_VERIFIED_NICKNAMES_A if side == 'A' else self.config.COL_VERIFIED_NICKNAMES_B
        val = row.get(col)
        return str(val) if val and not pd.isna(val) else None

    def _get_verified_entity_id(self, row: pd.Series, side: str) -> Optional[str]:
        col = self.config.COL_VERIFIED_ENTITY_ID_A if side == 'A' else self.config.COL_VERIFIED_ENTITY_ID_B
        val = row.get(col)
        return _coerce_entity_id(val)

    def _get_verified_status(self, row: pd.Series, side: str) -> Optional[str]:
        col = self.config.COL_VERIFIED_STATUS_A if side == 'A' else self.config.COL_VERIFIED_STATUS_B
        val = row.get(col)
        if val and not pd.isna(val):
            s = str(val).strip()
            return s if s and s.lower() not in ('nan', 'none') else None
        return None

    def _get_verified_id_number(self, row: pd.Series, side: str) -> Optional[str]:
        col = self.config.COL_VERIFIED_ID_NUMBER_A if side == 'A' else self.config.COL_VERIFIED_ID_NUMBER_B
        val = row.get(col)
        return _coerce_entity_id(val)

    def _get_phone(self, row: pd.Series, side: str) -> str:
        col = self.config.COL_PHONE_A if side == 'A' else self.config.COL_PHONE_B
        val = row.get(col)
        # Normalize phone to consistent format (prevents graph splits from formatting)
        return normalize_phone(str(val)) if val and not pd.isna(val) else ''

    def _get_other_phone(self, row: pd.Series, side: str) -> str:
        col = self.config.COL_PHONE_B if side == 'A' else self.config.COL_PHONE_A
        val = row.get(col)
        # Normalize phone to consistent format
        return normalize_phone(str(val)) if val and not pd.isna(val) else ''


# ============================================================================
# VECTORIZATION
# ============================================================================

class EntityVectorizer:
    """TF-IDF vectorization for name mentions."""

    def __init__(self, config: Config):
        self.config = config
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.token_vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=lambda x: x.split(),
            token_pattern=None,
            min_df=1,
            max_df=0.90,
            sublinear_tf=True,
        )
        self.char_vectors: Optional[csr_matrix] = None
        self.token_vectors: Optional[csr_matrix] = None
        self.idf_weights: Dict[str, float] = {}
        self.mention_index: Dict[str, int] = {}
        self._fitted = False
        self._corpus_size = 0  # Fix 29.23: Track corpus size for IDF demotion threshold

    def fit_transform(self, mentions: List[NameMention]) -> None:
        """Build TF-IDF vectors for all mentions.

        Handles edge cases:
        - Single/few documents: Adjusts max_df to avoid ValueError when all terms
          appear in 100% of documents
        - Low vocabulary diversity: Falls back to vectorizer without max_df constraints
          when all terms get pruned
        """
        if not mentions:
            self._fitted = True
            return

        self.mention_index = {m.mention_id: i for i, m in enumerate(mentions)}
        normalized_texts = [m.normalized for m in mentions]
        token_texts = [' '.join(m.tokens) for m in mentions]

        n_docs = len(mentions)
        self._corpus_size = n_docs  # Fix 29.23: Store for IDF threshold adjustments

        # Bug fix 1: For single/few documents, max_df thresholds can cause failures
        # when all terms appear in 100% of documents. Use max_df=1.0 for small datasets.
        if n_docs <= 2:
            # With 1-2 documents, any max_df < 1.0 will prune all terms
            char_max_df = 1.0
            token_max_df = 1.0
        else:
            char_max_df = 0.95
            token_max_df = 0.90

        # Create char vectorizer with appropriate max_df
        char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=1,
            max_df=char_max_df,
            sublinear_tf=True,
        )

        try:
            self.char_vectors = char_vectorizer.fit_transform(normalized_texts)
        except ValueError:
            # Bug fix 2: Fallback if all terms get pruned (low vocabulary diversity)
            fallback_char_vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=1,
                max_df=1.0,  # No upper limit
                sublinear_tf=True,
            )
            self.char_vectors = fallback_char_vectorizer.fit_transform(normalized_texts)
            char_vectorizer = fallback_char_vectorizer

        # Store the actually used vectorizer for potential later use
        self.char_vectorizer = char_vectorizer

        if any(t.strip() for t in token_texts):
            # Create token vectorizer with appropriate max_df
            token_vectorizer = TfidfVectorizer(
                analyzer='word',
                tokenizer=lambda x: x.split(),
                token_pattern=None,
                min_df=1,
                max_df=token_max_df,
                sublinear_tf=True,
            )

            try:
                self.token_vectors = token_vectorizer.fit_transform(token_texts)
                feature_names = token_vectorizer.get_feature_names_out()
                idf_values = token_vectorizer.idf_
                self.idf_weights = dict(zip(feature_names, idf_values))
            except ValueError:
                # Bug fix 2: Fallback if all terms get pruned (e.g., same name on many phones)
                fallback_token_vectorizer = TfidfVectorizer(
                    analyzer='word',
                    tokenizer=lambda x: x.split(),
                    token_pattern=None,
                    min_df=1,
                    max_df=1.0,  # No upper limit - keep all terms
                    sublinear_tf=True,
                )
                self.token_vectors = fallback_token_vectorizer.fit_transform(token_texts)
                feature_names = fallback_token_vectorizer.get_feature_names_out()
                idf_values = fallback_token_vectorizer.idf_
                self.idf_weights = dict(zip(feature_names, idf_values))
                token_vectorizer = fallback_token_vectorizer

            # Store the actually used vectorizer
            self.token_vectorizer = token_vectorizer
        else:
            self.token_vectors = csr_matrix((len(mentions), 1))
            self.idf_weights = {}

        self._fitted = True

    def get_char_vector_by_idx(self, idx: int) -> Optional[csr_matrix]:
        if not self._fitted or self.char_vectors is None:
            return None
        if idx < 0 or idx >= self.char_vectors.shape[0]:
            return None
        return self.char_vectors[idx]

    def get_idf(self, token: str) -> float:
        # Fix 29.15: Return HIGH default IDF for unknown tokens, not 1.0.
        # Unknown tokens (not in cube1 corpus) are likely rare phonebook-only names.
        # Default 1.0 caused false demotions because IDF_DEMOTION_THRESHOLD is 1.5.
        # Common names would appear in corpus with actual IDF values; unknown = not common.
        # Using 5.0 (above 1.5 threshold) ensures unknown tokens don't trigger demotion.
        return self.idf_weights.get(token, 5.0)

    def get_index(self, mention_id: str) -> Optional[int]:
        return self.mention_index.get(mention_id)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def corpus_size(self) -> int:
        """Fix 29.23: Return corpus size for IDF threshold adjustments."""
        return self._corpus_size


# ============================================================================
# SIMILARITY SCORING
# ============================================================================

AL_PREFIX_PATTERN = re.compile(r'^אל')


class AmbiguityGate:
    """Block edges when 1-token mention maps to 2+ distinct head signatures."""

    def __init__(self, mentions: List[NameMention]):
        self.phone_token_signatures: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for m in mentions:
            if len(m.tokens) < 2:
                continue
            signature = self._get_head_signature(m)
            for token in m.tokens:
                # Fix 27.6: Apply phonetic normalization to token keys
                # This ensures phonetic variants (קאסם/כאסם, טאהר/תאהר) are grouped
                # together for ambiguity detection. Without this, the same name
                # with different transliteration spellings would be stored under
                # different keys, causing false "safe" declarations in is_safe_singleton.
                token_key = normalize_arabic_phonetic(token)
                self.phone_token_signatures[m.phone][token_key].add(signature)

    def _get_head_signature(self, m: NameMention) -> str:
        """Extract canonical head signature: family name + kunya if present."""
        tokens = m.tokens
        if not tokens:
            return ''

        # Filter out abbreviation tokens (containing ״ or ")
        # These are role markers like מ״פ, רס״ן, סא״ל that shouldn't be family names
        filtered = [t for t in tokens if '״' not in t and '"' not in t]

        # Use filtered list for family, fallback to original if all filtered
        family = filtered[-1] if filtered else tokens[-1]

        # Check for kunya (אבו = Abu) in original tokens
        # Fix 21.2: Only אבו (Abu - father of) is common in this context
        kunya = None
        for t in tokens:
            if t.startswith('אבו'):
                kunya = t
                break
        return f"{family}+{kunya}" if kunya else family

    def should_block_edge(self, m1: NameMention, m2: NameMention) -> bool:
        """Block if 1-token mention maps to 2+ distinct signatures.

        Fix 27.1: Signature-Aware Disambiguation - Don't block if the multi-token
        mention's signature is one of the known signatures for the ambiguous token.
        This allows "אבו-עאדף" to cluster with "אבו-עאדף אגס אלער׳נה" even when
        "אבו-עאדף" is ambiguous (appears with multiple families on same phone),
        because the full name provides disambiguation.
        """
        if m1.phone != m2.phone:
            return False
        for single, multi in [(m1, m2), (m2, m1)]:
            if len(single.tokens) == 1 and len(multi.tokens) > 1:
                token = single.tokens[0]
                # Fix 29.1: Apply phonetic normalization to lookup key to match indexing
                # The index stores tokens with normalize_arabic_phonetic() (line 1074),
                # so lookup must use the same normalization. Without this, phonetic
                # variants like קאסם/כאסם would be stored under 'כאסם' but looked up
                # under 'קאסם', causing false "safe" declarations.
                token_key = normalize_arabic_phonetic(token)
                signatures = self.phone_token_signatures[single.phone].get(token_key, set())
                if len(signatures) > 1:
                    # Fix 27.1: Check if multi-token provides disambiguation
                    multi_sig = self._get_head_signature(multi)
                    if multi_sig and multi_sig in signatures:
                        # Safe - the full name disambiguates which person this is
                        continue
                    return True
        return False


class PhoneAliasIndex:
    """Index of verified_name <-> nicknames from call data."""

    def __init__(self, mentions: List[NameMention], normalizer: Normalizer = None):
        self.index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.reverse_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # Use provided normalizer or create default
        # This ensures nickname normalization matches mention normalization
        self._normalizer = normalizer or Normalizer(Config())

        for m in mentions:
            if m.verified_name and m.verified_nicknames:
                for nick in _split_nicknames(m.verified_nicknames):
                    # Use same normalization as mention text for consistency
                    nick_norm = self._normalizer.normalize(nick)
                    if nick_norm:
                        self.index[m.phone][m.verified_name].add(nick_norm)
                        self.reverse_index[m.phone][nick_norm].add(m.verified_name)

    def get_verified_for_nickname(self, phone: str, mention_normalized: str, mention_tokens: List[str] = None) -> Optional[str]:
        """Return verified name if nickname matches (exact OR contained) (Fix 1.3).

        Args:
            phone: Phone number to search within
            mention_normalized: The normalized mention text
            mention_tokens: Optional list of tokens from the mention (for containment check)
        """
        # First try exact match (original behavior)
        matches = self.reverse_index[phone].get(mention_normalized, set())
        if len(matches) == 1:
            return list(matches)[0]

        # If mention_tokens provided, try containment matching
        if mention_tokens:
            mention_token_set = set(mention_tokens)

            for nickname_norm, verified_names in self.reverse_index[phone].items():
                if len(verified_names) != 1:
                    continue
                nickname_tokens = set(nickname_norm.split())

                # Guard: nickname must have ≥2 tokens for containment
                # (prevents common first names matching everything)
                if len(nickname_tokens) < 2:
                    continue

                # Containment check: nickname tokens subset of mention tokens
                if nickname_tokens.issubset(mention_token_set):
                    return list(verified_names)[0]

        return None


class SimilarityScorer:
    """Hybrid similarity scoring between name mentions."""

    def __init__(
        self,
        config: Config,
        vectorizer: EntityVectorizer,
        ambiguity_gate: Optional[AmbiguityGate] = None,
        phone_alias_index: Optional[PhoneAliasIndex] = None,
        normalizer: Optional['Normalizer'] = None
    ):
        self.config = config
        self.vectorizer = vectorizer
        self.ambiguity_gate = ambiguity_gate
        self.phone_alias_index = phone_alias_index
        self.normalizer = normalizer or Normalizer(config)
        self.noise_tokens = config.get_noise_tokens_normalized()

    def compute(
        self,
        mention1: NameMention,
        mention2: NameMention,
        idx1: int = None,
        idx2: int = None
    ) -> float:
        """Compute hybrid similarity between two mentions."""

        # BLMZ SAFETY: "unknown speaker" is a negative constraint.
        # Never link a BLMZ marker to a non-BLMZ mention (even if row-level IDs match).
        if mention1.is_blmz != mention2.is_blmz:
            return 0.0

        # Hard link / hard block based on verified_entity_id (strongest signal)
        # (But never use IDs as a linkage signal for BLMZ mentions.)
        if (mention1.verified_entity_id and mention2.verified_entity_id and
                (not mention1.is_blmz) and (not mention2.is_blmz)):
            if mention1.verified_entity_id == mention2.verified_entity_id:
                return 1.0
            else:
                return 0.0

        # Must-not-link check WITH same-field alias override (Fix 1.5)
        if mention2.mention_id in mention1.must_not_link:
            # Check if this is actually a same-field alias before blocking
            if self._are_same_field_aliases(mention1, mention2):
                return 0.92  # Override must_not_link for recognized aliases
            return 0.0

        # v8 Fix A: verified_name is strong evidence, NOT hard block
        # Allow containment (מחמד אלפג׳לה vs מחמד אלפג׳לה אבו-אלעבד) or high similarity
        if mention1.verified_name and mention2.verified_name:
            v1_norm = self.normalizer.normalize(mention1.verified_name)
            v2_norm = self.normalizer.normalize(mention2.verified_name)
            # Fix 22.3: Apply phonetic normalization for Arabic-Hebrew homophones
            # Without this, קאסם vs כאסם would be treated as different people
            v1_phonetic = normalize_arabic_phonetic(v1_norm)
            v2_phonetic = normalize_arabic_phonetic(v2_norm)
            if v1_phonetic != v2_phonetic:
                # NOT a hard block - check if containment or high similarity
                v1_tokens = set(normalize_arabic_phonetic(t) for t in self.normalizer.tokenize(v1_norm))
                v2_tokens = set(normalize_arabic_phonetic(t) for t in self.normalizer.tokenize(v2_norm))

                # Allow if one is subset of other (expansion/containment case)
                if v1_tokens <= v2_tokens or v2_tokens <= v1_tokens:
                    pass  # Continue to normal scoring
                # Allow if very high fuzzy similarity (variant spelling)
                elif _token_sort_ratio(v1_phonetic, v2_phonetic) >= 88:
                    pass  # Continue to normal scoring
                else:
                    return 0.0  # Only block when clearly different people

        # Verified containment check (Fix 1.1B)
        containment_score = self._verified_containment_score(mention1, mention2)
        if containment_score is not None:
            return containment_score

        # Same-field alias check BEFORE AmbiguityGate (Refinement A)
        # This ensures alias pairs link even when not in must_not_link
        if self._are_same_field_aliases(mention1, mention2):
            return 0.92

        # AmbiguityGate - block edges for ambiguous 1-token bridges
        if self.ambiguity_gate and self.ambiguity_gate.should_block_edge(mention1, mention2):
            return 0.0

        # Call-level nickname matching (Fix 1.3b: pass mention_tokens)
        if self.phone_alias_index and mention1.phone == mention2.phone:
            if mention1.verified_name:
                verified_for_m2 = self.phone_alias_index.get_verified_for_nickname(
                    mention1.phone, mention2.normalized, mention2.tokens
                )
                if verified_for_m2 == mention1.verified_name:
                    return 0.92
            if mention2.verified_name:
                verified_for_m1 = self.phone_alias_index.get_verified_for_nickname(
                    mention2.phone, mention1.normalized, mention1.tokens
                )
                if verified_for_m1 == mention2.verified_name:
                    return 0.92

        # Arabic name pattern match
        arabic_match = self._arabic_name_match(
            mention1.normalized, mention2.normalized,
            mention1.tokens, mention2.tokens
        )
        if arabic_match >= 0.85:
            return arabic_match

        # Get matrix indices
        if idx1 is None:
            idx1 = self.vectorizer.get_index(mention1.mention_id)
        if idx2 is None:
            idx2 = self.vectorizer.get_index(mention2.mention_id)

        # Compute similarities
        char_sim = self._char_ngram_similarity(idx1, idx2)
        token_sim = self._token_set_similarity(mention1.normalized, mention2.normalized)
        idf_sim = self._idf_jaccard(mention1.tokens, mention2.tokens)

        # Weighted combination
        similarity = (
            self.config.WEIGHT_CHAR_NGRAM * char_sim +
            self.config.WEIGHT_TOKEN_SET * token_sim +
            self.config.WEIGHT_IDF_JACCARD * idf_sim
        )

        if arabic_match > 0:
            similarity = max(similarity, arabic_match * 0.9)

        # Similarity cap for 1-token <-> multi-token bridges
        len1 = len(mention1.tokens)
        len2 = len(mention2.tokens)
        if (len1 == 1 and len2 > 1) or (len2 == 1 and len1 > 1):
            # If the single-token is unambiguous on this phone, allow a slightly higher cap.
            cap = 0.65
            try:
                if self.ambiguity_gate:
                    single = mention1 if len1 == 1 else mention2
                    multi = mention2 if len1 == 1 else mention1
                    # Fix 29.1b: Apply phonetic normalization to lookup key to match indexing
                    # The index stores tokens normalized (line 1074), so lookups must match
                    single_token_key = normalize_arabic_phonetic(single.tokens[0])
                    sigs = self.ambiguity_gate.phone_token_signatures.get(single.phone, {}).get(single_token_key, set())
                    if len(sigs) <= 1:
                        cap = 0.80
                    # Fix 27.1: Signature-Aware Disambiguation
                    # When the multi-token mention's signature is one of the known signatures
                    # for the ambiguous single token, this is a SAFE edge because the full
                    # name provides disambiguation.
                    # Example: "אבו-עאדף" is ambiguous (maps to multiple families), but
                    # "אבו-עאדף אגס אלער׳נה" clearly identifies the אלער׳נה family.
                    elif len(sigs) > 1:
                        multi_sig = self.ambiguity_gate._get_head_signature(multi)
                        if multi_sig and multi_sig in sigs:
                            cap = 0.80  # Safe - full name disambiguates
            except (KeyError, IndexError, AttributeError) as e:
                # Log error but use conservative cap (0.65) as fallback.
                # These exceptions can occur if:
                # - KeyError: phone_token_signatures missing expected keys
                # - IndexError: single.tokens is unexpectedly empty
                # - AttributeError: ambiguity_gate structure is malformed
                logging.warning(
                    "Singleton cap calculation failed for mentions %s <-> %s: %s. "
                    "Using conservative cap=0.65.",
                    mention1.mention_id, mention2.mention_id, e
                )
            similarity = min(similarity, cap)

        return min(1.0, max(0.0, similarity))

    def _char_ngram_similarity(self, idx1: int, idx2: int) -> float:
        if idx1 is None or idx2 is None:
            return 0.0
        vec1 = self.vectorizer.get_char_vector_by_idx(idx1)
        vec2 = self.vectorizer.get_char_vector_by_idx(idx2)
        if vec1 is None or vec2 is None:
            return 0.0
        sim = cosine_similarity(vec1, vec2)[0, 0]
        return float(sim)

    def _token_set_similarity(self, text1: str, text2: str) -> float:
        """Token set similarity with noise filtering (Fix 1.2).

        Fix 24.4: Apply phonetic normalization before comparison to ensure
        geresh variants (ג׳ vs ג) and homophones (ק vs כ) are matched correctly.
        """
        if not text1 or not text2:
            return 0.0
        # Filter noise tokens before comparison
        tokens1 = [t for t in text1.split() if t not in self.noise_tokens]
        tokens2 = [t for t in text2.split() if t not in self.noise_tokens]
        if not tokens1 or not tokens2:
            return 0.0
        # Fix 24.4: Apply phonetic normalization to handle geresh and homophones
        clean1 = ' '.join(normalize_arabic_phonetic(t) for t in tokens1)
        clean2 = ' '.join(normalize_arabic_phonetic(t) for t in tokens2)
        return _token_set_ratio(clean1, clean2) / 100.0

    def _idf_jaccard(self, tokens1: List[str], tokens2: List[str]) -> float:
        """IDF-weighted Jaccard with FUZZY token matching (Fix 3.1).

        Instead of requiring exact token equality, allows fuzzy matching
        for tokens >= 3 chars with similarity >= 85%.

        Fix 24.4: Apply phonetic normalization before comparison to ensure
        geresh variants (ג׳ vs ג) and homophones (ק vs כ) are matched correctly.
        """
        # Filter noise tokens
        set1 = set(tokens1) - self.noise_tokens
        set2 = set(tokens2) - self.noise_tokens
        if not set1 or not set2:
            return 0.0

        # Build soft matches: for each token in set1, find best match in set2
        matched_pairs = []
        used_in_set2 = set()

        for t1 in set1:
            best_match = None
            best_score = 0
            # Fix 24.4: Apply phonetic normalization for comparison
            t1_phonetic = normalize_arabic_phonetic(t1)
            for t2 in set2:
                if t2 in used_in_set2:
                    continue
                t2_phonetic = normalize_arabic_phonetic(t2)
                # Exact match after phonetic normalization - always prefer
                if t1_phonetic == t2_phonetic:
                    best_match = t2
                    best_score = 100
                    break
                # Fuzzy match for longer tokens (≥3 chars)
                if len(t1_phonetic) >= 3 and len(t2_phonetic) >= 3:
                    score = _char_ratio(t1_phonetic, t2_phonetic)
                    if score >= 85 and score > best_score:  # 85% threshold (aggressive)
                        best_match = t2
                        best_score = score

            if best_match:
                matched_pairs.append((t1, best_match, best_score / 100.0))
                used_in_set2.add(best_match)

        # Debug: log near-miss token pairs (potential transliteration variants)
        if self.config.DEBUG_TOKEN_MATCHING:
            for t1 in set1:
                for t2 in set2:
                    if t1 != t2 and len(t1) >= 3 and len(t2) >= 3:
                        # Fix 24.4: Use phonetic normalization for debug logging too
                        t1_p = normalize_arabic_phonetic(t1)
                        t2_p = normalize_arabic_phonetic(t2)
                        ratio = _char_ratio(t1_p, t2_p)
                        if self.config.DEBUG_TOKEN_MATCHING_THRESHOLD <= ratio < 85:
                            print(f"[TOKEN_NEAR_MISS] '{t1}' vs '{t2}': {ratio}% (needs 85%)")

        if not matched_pairs:
            return 0.0

        # Compute weighted intersection (fuzzy matches weighted by match quality)
        inter_weight = sum(
            self.vectorizer.get_idf(t1) * match_score
            for t1, t2, match_score in matched_pairs
        )
        union_weight = sum(self.vectorizer.get_idf(t) for t in set1 | set2)

        return inter_weight / union_weight if union_weight > 0 else 0.0

    def _get_dynamic_threshold(self, m1: NameMention, m2: NameMention) -> float:
        """Get similarity threshold based on evidence strength (Fix 3.2).

        Aggressive settings - tune up if too many false merges.
        """
        len1 = len([t for t in m1.tokens if t not in self.noise_tokens])
        len2 = len([t for t in m2.tokens if t not in self.noise_tokens])
        min_len = min(len1, len2)

        # Base threshold (aggressive)
        threshold = 0.65  # Start lower than default 0.70

        # Lower threshold for longer names (more evidence)
        if min_len >= 4:
            threshold = 0.58  # 4+ tokens = very distinctive, can be more permissive
        elif min_len >= 3:
            threshold = 0.60  # 3 tokens = more distinctive

        # Lower threshold if one side has verified anchor
        if m1.verified_name or m2.verified_name:
            threshold = min(threshold, 0.58)

        # Keep stricter for short 2-token names (higher risk)
        if min_len <= 2:
            threshold = max(threshold, 0.65)

        return threshold

    def _arabic_name_match(
        self,
        text1: str,
        text2: str,
        tokens1: List[str],
        tokens2: List[str]
    ) -> float:
        if not text1 or not text2:
            return 0.0

        # Fix 18.1: Apply phonetic normalization for Arabic-Hebrew homophones
        # קאסם vs כאסם (Qassem) should match as the SAME person.
        # Without this, they only get ~75% similarity due to ק/כ difference.
        text1_phonetic = normalize_arabic_phonetic(text1)
        text2_phonetic = normalize_arabic_phonetic(text2)
        tokens1_phonetic = [normalize_arabic_phonetic(t) for t in tokens1] if tokens1 else []
        tokens2_phonetic = [normalize_arabic_phonetic(t) for t in tokens2] if tokens2 else []

        if text1_phonetic == text2_phonetic:
            return 1.0

        # Check אל- prefix match (using phonetic-normalized text)
        # IMPORTANT: Use .strip() to handle leading space left when normalizer
        # converts "אל-חמזה" to "אל חמזה" - regex strips "אל" but leaves " חמזה"
        t1_stripped = AL_PREFIX_PATTERN.sub('', text1_phonetic).strip()
        t2_stripped = AL_PREFIX_PATTERN.sub('', text2_phonetic).strip()

        if t1_stripped == t2_stripped and t1_stripped:
            return 0.95

        if len(text1_phonetic) > len(text2_phonetic) and text1_phonetic.endswith(text2_phonetic):
            prefix = text1_phonetic[:-len(text2_phonetic)]
            if prefix in ('אל', 'ال'):
                return 0.95
            if len(text2_phonetic) >= 3:
                return 0.80

        if len(text2_phonetic) > len(text1_phonetic) and text2_phonetic.endswith(text1_phonetic):
            prefix = text2_phonetic[:-len(text1_phonetic)]
            if prefix in ('אל', 'ال'):
                return 0.95
            if len(text1_phonetic) >= 3:
                return 0.80

        # Token-level check (using phonetic-normalized tokens)
        if tokens1_phonetic and tokens2_phonetic:
            stripped1 = [AL_PREFIX_PATTERN.sub('', t) for t in tokens1_phonetic]
            stripped2 = [AL_PREFIX_PATTERN.sub('', t) for t in tokens2_phonetic]
            stripped1 = [t for t in stripped1 if t]
            stripped2 = [t for t in stripped2 if t]
            if stripped1 and stripped2:
                set1 = set(stripped1)
                set2 = set(stripped2)
                if set1 == set2:
                    return 0.90

        return 0.0

    def _are_same_field_aliases(self, m1: NameMention, m2: NameMention) -> bool:
        """Check if mentions are aliases from the same + field (Fix 1.5)."""
        # Must be from same call + same side + same original_field
        if m1.call_id != m2.call_id or m1.side != m2.side:
            return False
        if m1.original_field != m2.original_field:
            return False
        # Both must be from a multi-segment field
        if m1.segment_count < 2 or m2.segment_count < 2:
            return False

        set1 = set(m1.tokens) - self.noise_tokens
        set2 = set(m2.tokens) - self.noise_tokens
        list1 = [t for t in m1.tokens if t not in self.noise_tokens]
        list2 = [t for t in m2.tokens if t not in self.noise_tokens]

        if not set1 or not set2:
            return False

        shared = set1 & set2

        # Rule 1: ≥2 shared meaningful tokens
        if len(shared) >= 2:
            return True

        # Rule 2: Full containment with ≥2 tokens in shorter
        shorter_set, longer_set = (set1, set2) if len(set1) <= len(set2) else (set2, set1)
        if shorter_set.issubset(longer_set) and len(shorter_set) >= 2:
            return True

        # Rule 3: 1-token prefix expansion (משה ↔ משה אלקנה)
        # Single token equals the FIRST token of the longer name
        # Refinement B: Only allow if single token is NOT ambiguous on that phone
        shorter_list, longer_list = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
        if len(shorter_list) == 1 and len(longer_list) >= 2:
            if shorter_list[0] == longer_list[0]:  # First name match
                # Check ambiguity guard (Refinement B)
                single_token = shorter_list[0]
                if self.ambiguity_gate:
                    # Fix 29.1c: Apply phonetic normalization to lookup key to match indexing
                    single_token_key = normalize_arabic_phonetic(single_token)
                    signatures = self.ambiguity_gate.phone_token_signatures[m1.phone].get(single_token_key, set())
                    if len(signatures) > 1:
                        # Token is ambiguous - don't allow 1-token prefix expansion
                        return False
                return True

        return False

    def _verified_containment_score(self, m1: NameMention, m2: NameMention) -> Optional[float]:
        """Check if verified name tokens are contained in other mention (Fix 1.1B)."""
        verified, other = (m1, m2) if m1.verified_name else (m2, m1) if m2.verified_name else (None, None)
        if not verified:
            return None

        # Normalize verified_name (stored raw in v7) using same normalizer as mentions
        verified_name_normalized = self.normalizer.normalize(verified.verified_name)
        # Fix 23.1: Apply phonetic normalization for Arabic-Hebrew homophones consistency
        # Without this, קאסם vs כאסם would fail containment check
        verified_tokens = set(normalize_arabic_phonetic(t) for t in self.normalizer.tokenize(verified_name_normalized)) - self.noise_tokens
        other_tokens = set(normalize_arabic_phonetic(t) for t in other.tokens) - self.noise_tokens

        # Guard 1: verified must have ≥2 meaningful tokens
        if len(verified_tokens) < 2:
            return None

        # Guard 2: other must also have ≥2 meaningful tokens (prevents garbage merges)
        if len(other_tokens) < 2:
            return None

        if verified_tokens.issubset(other_tokens):
            return 0.92
        return None


# ============================================================================
# GRAPH BUILDING
# ============================================================================


@dataclass
class IdentitySignature:
    """Compact identity evidence for cross-phone linking.

    We intentionally treat phone as a *soft prior* (thresholding/guards), not a hard boundary.
    This signature aggregates all name evidence for a cluster so we can compare
    ANY name variant ↔ ANY name variant across phones.
    """
    cluster_id: str
    phone: str
    resolution_type: str

    # Entity ID for hard linking (highest priority)
    verified_entity_id: Optional[str] = None
    phonebook_quality: str = ''  # LOW / MED / HIGH (v9)

    # Strong-ish evidence
    verified_names: Set[str] = field(default_factory=set)
    verified_nicknames: Set[str] = field(default_factory=set)

    # Weak-ish evidence
    canonical_name: str = ""
    nicknames: Set[str] = field(default_factory=set)
    top_mentions: Set[str] = field(default_factory=set)

    # Derived for matching / blocking
    all_names_normalized: Set[str] = field(default_factory=set)
    # API-ready names (light normalization — no kunya transforms)
    api_name: str = ""
    all_names_api: Set[str] = field(default_factory=set)
    first_tokens: Set[str] = field(default_factory=set)
    last_tokens: Set[str] = field(default_factory=set)
    max_token_len: int = 0
    def anchor_level(self) -> str:
        """Return anchor strength: 'strong', 'semi', or 'none'.

        v9 refinement:
        - CALL_VERIFIED / verified_names / verified_entity_id are strong anchors.
        - PHONEBOOK is only a strong anchor when the phonebook contact is high-quality
          (to avoid over-merging on generic labels like 'אמא', 'עבודה', 'מונית').
        """
        # Hard anchor: explicit global entity_id
        if self.verified_entity_id:
            return 'strong'

        # Verified names from call data
        if self.resolution_type == 'CALL_VERIFIED' or bool(self.verified_names):
            return 'strong'

        # Phonebook anchors are quality-gated
        if self.resolution_type == 'PHONEBOOK':
            if (self.phonebook_quality or '') == 'HIGH':
                return 'strong'
            if (self.phonebook_quality or '') == 'MED':
                return 'semi'
            return 'none'

        # Verified nicknames only: semi-anchor
        if bool(self.verified_nicknames):
            return 'semi'

        return 'none'

    def is_anchor(self) -> bool:
        """Anchors (strong or semi) can 'pull' other clusters across phones."""
        return self.anchor_level() in ("strong", "semi")


@dataclass
class EvidenceBundle:
    """Bundle of name evidence for Yanis API scoring (Fix 30.1 / 30.2).

    Separates identity-defining names (strong) from auxiliary names (weak)
    to enable principled aggregation of API pair scores.
    """
    cluster_id: str
    phone: str
    strong_names_api: List[str]   # Identity-defining (deduped, longest-first)
    weak_names_api: List[str]     # Nicknames, kunyas, stripped variants
    all_names_api: List[str]      # Combined for discovery
    has_verified_name: bool
    has_entity_id: bool
    resolution_type: str
    anchor_level: str             # 'strong', 'semi', 'none'


class SimilarityGraph:
    """Build per-phone similarity graphs for clustering."""

    def __init__(self, config: Config, vectorizer: EntityVectorizer, scorer: SimilarityScorer):
        self.config = config
        self.vectorizer = vectorizer
        self.scorer = scorer

    def build_phone_graphs(self, mentions: List[NameMention]) -> Dict[str, nx.Graph]:
        """Build similarity graphs grouped by phone number."""
        by_phone: Dict[str, List[Tuple[NameMention, int]]] = defaultdict(list)

        for m in mentions:
            idx = self.vectorizer.get_index(m.mention_id)
            if idx is not None:
                by_phone[m.phone].append((m, idx))

        graphs = {}
        for phone in sorted(by_phone.keys()):
            mention_list = by_phone[phone]
            phone_mentions = [m for m, _ in mention_list]
            phone_indices = [idx for _, idx in mention_list]
            graphs[phone] = self._build_graph(phone_mentions, phone_indices)

        return graphs

    def _build_graph(self, mentions: List[NameMention], indices: List[int]) -> nx.Graph:
        """Build similarity graph for a set of mentions."""
        G = nx.Graph()

        for m in mentions:
            G.add_node(m.mention_id, mention=m)

        n = len(mentions)
        for i in range(n):
            for j in range(i + 1, n):
                # NOTE: do not hard-skip must_not_link pairs here.
                # SimilarityScorer.compute() contains override logic for alias segments.

                sim = self.scorer.compute(mentions[i], mentions[j], indices[i], indices[j])

                # Use dynamic threshold based on name length and verified status.
                # Long names (4+ tokens) get lower threshold (0.58) since they have more
                # distinctive information. Verified anchors also get lower threshold.
                # This activates the previously-dead _get_dynamic_threshold logic.
                dynamic_thresh = self.scorer._get_dynamic_threshold(mentions[i], mentions[j])
                if sim >= dynamic_thresh:
                    G.add_edge(mentions[i].mention_id, mentions[j].mention_id, weight=sim)

        return G


# ============================================================================
# CHINESE WHISPERS CLUSTERING
# ============================================================================

class ChineseWhispers:
    """Chinese Whispers clustering algorithm."""

    def __init__(self, config: Config):
        self.config = config
        self.max_iterations = config.CW_MAX_ITERATIONS
        self.seed = config.CW_SEED

    def cluster_with_constraints(
        self,
        graph: nx.Graph,
        mentions: List[NameMention]
    ) -> Dict[str, str]:
        """Cluster with verified anchors and בלמ״ז isolation."""
        mention_by_id = {m.mention_id: m for m in mentions}
        labels = self._cluster_with_verified_anchors(graph, mentions)
        labels = self._isolate_blmz(labels, mentions)
        labels = self._split_verified_conflicts(labels, mention_by_id)
        return labels

    def _cluster_with_verified_anchors(
        self,
        graph: nx.Graph,
        mentions: List[NameMention]
    ) -> Dict[str, str]:
        """Run Chinese Whispers with verified mentions as fixed anchors."""
        if graph.number_of_nodes() == 0:
            return {}

        rng = random.Random(self.seed)

        # Build: (phone, verified_name) -> anchor_id
        verified_anchors: Dict[Tuple[str, str], str] = {}
        anchor_counter = 0

        for m in mentions:
            if m.verified_name and m.mention_id in graph.nodes():
                key = (m.phone, m.verified_name)
                if key not in verified_anchors:
                    verified_anchors[key] = f"verified_{anchor_counter}"
                    anchor_counter += 1

        mention_by_id = {m.mention_id: m for m in mentions}

        # Initialize labels
        labels = {}
        verified_mention_ids: Set[str] = set()

        for i, node in enumerate(graph.nodes()):
            node_mention = mention_by_id.get(node)
            if node_mention and node_mention.verified_name:
                key = (node_mention.phone, node_mention.verified_name)
                labels[node] = verified_anchors[key]
                verified_mention_ids.add(node)
            else:
                labels[node] = f"c_{i}"

        # CW iterations - skip verified mentions
        for iteration in range(self.max_iterations):
            nodes = list(graph.nodes())
            rng.shuffle(nodes)
            changed = False

            for node in nodes:
                if graph.degree(node) == 0:
                    continue
                if node in verified_mention_ids:
                    continue

                neighbor_votes = Counter()
                for neighbor in graph.neighbors(node):
                    weight = graph[node][neighbor].get('weight', 1.0)
                    neighbor_votes[labels[neighbor]] += weight

                if neighbor_votes:
                    # Deterministic tie-breaking: when multiple labels have
                    # the same max vote, pick lexicographically smallest
                    candidates = neighbor_votes.most_common()
                    max_vote = candidates[0][1]
                    tied_labels = [label for label, vote in candidates if vote == max_vote]
                    best_label = min(tied_labels)  # Deterministic tie-break

                    if labels[node] != best_label:
                        labels[node] = best_label
                        changed = True

            if not changed:
                break

        return labels

    def _isolate_blmz(
        self,
        labels: Dict[str, str],
        mentions: List[NameMention]
    ) -> Dict[str, str]:
        """Ensure each בלמ״ז mention gets its own cluster."""
        blmz_counter = 0
        for m in mentions:
            if m.is_blmz and m.mention_id in labels:
                labels[m.mention_id] = f"blmz_{blmz_counter}"
                blmz_counter += 1
        return labels

    def _split_verified_conflicts(
        self,
        labels: Dict[str, str],
        mention_by_id: Dict[str, NameMention]
    ) -> Dict[str, str]:
        """Split clusters that contain conflicting verified names.

        Safety-first behavior:
        - Each distinct verified_name in a conflicted cluster gets its own explicit label.
        - All unverified mentions in that conflicted cluster are quarantined into an
          "{cluster_id}_unresolved" bucket rather than being implicitly assigned to
          the first verified name encountered.

        This prevents corrupting ground-truth by "stealing" ambiguous mentions.
        """
        cluster_to_mentions: Dict[str, List[str]] = defaultdict(list)
        for mid, cid in labels.items():
            cluster_to_mentions[cid].append(mid)

        def _safe_suffix(name: str) -> str:
            # Unicode-safe and collision-resistant suffix.
            # Hebrew/Arabic names would otherwise collapse under ASCII-only slugging.
            raw = (name or "").strip()
            norm = unicodedata.normalize("NFKC", raw)
            base = re.sub(r"[^\w]+", "_", norm, flags=re.UNICODE).strip("_")
            if not base:
                base = "name"
            digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
            return f"{base[:16]}_{digest}"

        new_labels = labels.copy()

        for cluster_id, mention_ids in cluster_to_mentions.items():
            verified_groups: Dict[str, List[str]] = defaultdict(list)
            verified_mids: Set[str] = set()

            for mid in mention_ids:
                m = mention_by_id.get(mid)
                if m and m.verified_name:
                    verified_groups[m.verified_name].append(mid)
                    verified_mids.add(mid)

            # No conflict if 0 or 1 verified names present.
            if len(verified_groups) <= 1:
                continue

            # --- CONFLICT DETECTED ---
            # 1) Isolate each verified group into a deterministic cluster ID.
            for vname, mids in verified_groups.items():
                new_cid = f"{cluster_id}_v_{_safe_suffix(vname)}"
                for mid in mids:
                    new_labels[mid] = new_cid

            # 2) Quarantine all unverified mentions so they are not falsely attributed.
            unresolved_cid = f"{cluster_id}_unresolved"
            for mid in mention_ids:
                if mid not in verified_mids:
                    new_labels[mid] = unresolved_cid

        return new_labels


class Cube2Matcher:
    """Match clusters/mentions to cube2 contacts and expose contact metadata.

    v9: cube2 is treated as evidence (not just a post-hoc label source).
    We keep cube2 *out* of mention-level clustering (to avoid over-merging),
    but we surface stable contact keys + quality tiers so the resolver can
    safely bridge fragmented clusters within a phone (Stage 6.25).
    """

    QUALITY_RANK: Dict[str, int] = {'LOW': 0, 'MED': 1, 'HIGH': 2}

    def __init__(self, config: Config, cube2_df: pd.DataFrame = None):
        self.config = config
        self.normalizer = Normalizer(config)

        # Precompute token sets in normalized space for consistent gating
        self.noise_tokens: Set[str] = config.get_noise_tokens_normalized()
        self.generic_tokens: Set[str] = {
            self.normalizer.normalize(t) for t in (config.CUBE2_GENERIC_TOKENS or set())
        }

        self.contacts_by_phone: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Fix 25.1: Global name index for cross-phone phonebook matching
        # Maps normalized_name -> list of contacts (regardless of phone)
        self.global_name_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Fix 29.28: Track nickname frequency for unique nickname bonus
        # Maps normalized_nickname -> set of entity_ids that have this nickname
        # A nickname is "unique" if only one entity has it (len == 1)
        self.nickname_entity_ids: Dict[str, Set[str]] = defaultdict(set)
        self._index_contacts(cube2_df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def finalize_contact_quality(self, vectorizer: Optional['EntityVectorizer'] = None) -> None:
        """Optionally enrich contacts with corpus stats once a vectorizer exists.

        We do NOT use IDF to *demote* real names (many valid Hebrew names are common),
        but we store it for observability and potential future tuning.

        Fix 29.23: For small corpora (< MIN_CORPUS_FOR_IDF_DEMOTION), skip actual IDF
        computation and use default 5.0 for all contacts. This prevents false demotions
        caused by artificially low IDF values in small datasets.
        """
        if not self.contacts_by_phone:
            return

        # Fix 29.23: Skip IDF computation for small corpora.
        # log(N/df) produces artificially low values when N is small, causing
        # rare names to be incorrectly demoted. Default to 5.0 (above threshold).
        use_actual_idf = (
            vectorizer is not None
            and vectorizer.corpus_size >= self.config.MIN_CORPUS_FOR_IDF_DEMOTION
        )

        for _, contacts in self.contacts_by_phone.items():
            for c in contacts:
                tokens = c.get('tokens') or []
                meaningful = [
                    t for t in tokens
                    if t and t not in self.noise_tokens and t not in self.generic_tokens
                ]
                if use_actual_idf and meaningful:
                    idfs = [vectorizer.get_idf(t) for t in meaningful]
                    c['mean_idf'] = float(sum(idfs) / max(len(idfs), 1))
                else:
                    # Fix 16.5 + Fix 29.15 + Fix 29.23: Default to 5.0 when:
                    # - IDF data unavailable (no vectorizer or empty tokens)
                    # - Corpus too small (< MIN_CORPUS_FOR_IDF_DEMOTION)
                    # 5.0 = "unknown/small corpus, assume rare" → won't trigger demotion.
                    c.setdefault('mean_idf', 5.0)

    def match(self, phone: str, mentions: List[NameMention]) -> Optional[Cube2Match]:
        """Match a set of mentions to the best cube2 contact for the given phone."""
        phone = normalize_phone(phone)
        if not phone:
            return None

        contacts = self.contacts_by_phone.get(phone, [])
        if not contacts:
            return None

        mention_texts = [m.normalized for m in mentions if m.normalized]
        if not mention_texts:
            return None

        best_contact = None
        best_score = 0.0
        second_score = 0.0
        best_nickname = None

        for contact in contacts:
            score, nickname = self._score_contact(contact, mention_texts)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_contact = contact
                best_nickname = nickname
            elif score > second_score:
                second_score = score

        if best_contact is None:
            return None

        if best_score < self.config.CUBE2_MATCH_THRESHOLD:
            return None

        margin = best_score - second_score
        # Fix 29.20: Cap scores at 1.0 for external API (internal sorting uses uncapped)
        capped_best = min(1.0, best_score)
        capped_second = min(1.0, second_score)
        return Cube2Match(
            name=best_contact['name'],
            nickname=best_nickname,
            score=capped_best,
            second_score=capped_second,
            margin=margin,  # Keep uncapped margin for confident gate
            entity_id=best_contact.get('entity_id'),
            contact_key=best_contact.get('contact_key'),
            name_normalized=best_contact.get('name_normalized', ''),
            tokens=list(best_contact.get('tokens') or []),
            quality_tier=best_contact.get('quality_tier', ''),
            quality_score=float(best_contact.get('quality_score') or 0.0),
            status=best_contact.get('status'),
            id_number=best_contact.get('id_number'),
        )

    def get_contact_by_key(self, phone: str, contact_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve full contact data by contact_key for nickname propagation."""
        if not contact_key:
            return None
        phone = normalize_phone(phone)
        if not phone:
            return None
        for contact in self.contacts_by_phone.get(phone, []):
            if contact.get('contact_key') == contact_key:
                return contact
        return None

    def match_global(self, mentions: List[NameMention]) -> Optional[Cube2Match]:
        """Search entire phonebook by name, ignoring phone number (Fix 25.1).

        This is a fallback for when per-phone lookup fails. It enables matching
        phonebook entries even when the cluster is on a different phone than
        the phonebook entry.

        Safety: Uses a higher threshold (0.90) and prefers contacts with entity_id
        to prevent false merges from common names like "David" or "Mohammed".
        """
        if not self.global_name_index:
            return None

        mention_texts = [m.normalized for m in mentions if m.normalized]
        if not mention_texts:
            return None

        best_contact = None
        best_score = 0.0
        second_score = 0.0
        best_nickname = None

        # Collect all candidate contacts from global index
        seen_contact_keys: Set[str] = set()
        candidate_contacts: List[Dict[str, Any]] = []

        for mention_text in mention_texts:
            # Fix 25.2: Safety - skip generic tokens for global search
            # "Taxi", "Mom", etc. could match many unrelated contacts globally
            # Per-phone matching is fine for these, but global is too risky
            if mention_text in self.generic_tokens:
                continue

            # Apply phonetic normalization for lookup
            mention_phonetic = normalize_arabic_phonetic(mention_text)

            # Try exact full-string match first
            if mention_text in self.global_name_index:
                for contact in self.global_name_index[mention_text]:
                    key = contact.get('contact_key', '')
                    if key not in seen_contact_keys:
                        seen_contact_keys.add(key)
                        candidate_contacts.append(contact)

            # Also try phonetic-normalized full-string lookup
            # This now works because we index phonetic keys in _index_contacts (Fix 25.2)
            if mention_phonetic != mention_text and mention_phonetic in self.global_name_index:
                for contact in self.global_name_index[mention_phonetic]:
                    key = contact.get('contact_key', '')
                    if key not in seen_contact_keys:
                        seen_contact_keys.add(key)
                        candidate_contacts.append(contact)

            # Fix 28.3 (Finding 47): Also try hyphen-normalized full-string lookup
            # Names like "עבד-אל-דימאן" may be in cluster but indexed as "עבד אל דימאן" (spaces)
            # Try both directions: hyphen-to-space and space-to-hyphen variants
            mention_hyphen_to_space = mention_text.replace('-', ' ')
            mention_hyphen_to_space = ' '.join(mention_hyphen_to_space.split())  # collapse multi-spaces
            if mention_hyphen_to_space != mention_text and mention_hyphen_to_space in self.global_name_index:
                for contact in self.global_name_index[mention_hyphen_to_space]:
                    key = contact.get('contact_key', '')
                    if key not in seen_contact_keys:
                        seen_contact_keys.add(key)
                        candidate_contacts.append(contact)

            # Fix 26.3: Also try TOKEN-based lookup for fuzzy matching
            # This enables finding contacts when full-string differs slightly
            # (e.g., "עבד-אלדימאן מידאוי" vs "עבד-אלדימן מידאוי")
            # The shared token "מידאוי" will find the candidate.
            mention_tokens = self.normalizer.tokenize(mention_text)
            # Expert Fix 2: Expand hyphenated tokens so "אבו-אחמד" also looks up
            # "אבו" and "אחמד" as separate token keys in the index.
            expanded_tokens = []
            for t in mention_tokens:
                expanded_tokens.append(t)
                if '-' in t:
                    expanded_tokens.extend(p for p in t.split('-') if p)
            mention_tokens = expanded_tokens
            for tok in mention_tokens:
                if not tok or tok in self.generic_tokens or tok in self.noise_tokens:
                    continue
                if len(tok) < 3:  # Skip very short tokens
                    continue
                # Lookup by token
                if tok in self.global_name_index:
                    for contact in self.global_name_index[tok]:
                        key = contact.get('contact_key', '')
                        if key not in seen_contact_keys:
                            seen_contact_keys.add(key)
                            candidate_contacts.append(contact)
                # Also try phonetic token
                tok_phonetic = normalize_arabic_phonetic(tok)
                if tok_phonetic and tok_phonetic != tok and tok_phonetic in self.global_name_index:
                    for contact in self.global_name_index[tok_phonetic]:
                        key = contact.get('contact_key', '')
                        if key not in seen_contact_keys:
                            seen_contact_keys.add(key)
                            candidate_contacts.append(contact)

        # Fix 28.1 (Finding 23): Smart candidate selection when too many candidates
        # Instead of hard-aborting when >100 candidates, sort by relevance and take top 100.
        # Prioritize: (1) entity_id presence, (2) token overlap, (3) quality tier
        # This ensures contacts with matching rare tokens or entity_id are always scored.
        MAX_GLOBAL_CANDIDATES = 100
        if len(candidate_contacts) > MAX_GLOBAL_CANDIDATES:
            # Compute relevance scores for prioritization (not full scoring - just quick heuristics)
            mention_tokens_set = set()
            for txt in mention_texts:
                mention_tokens_set.update(txt.split())
            # Apply phonetic normalization to mention tokens for matching
            mention_tokens_phonetic = {normalize_arabic_phonetic(t) for t in mention_tokens_set}

            def candidate_priority(contact: Dict[str, Any]) -> Tuple[int, int, int, float]:
                """Compute priority tuple for sorting (higher = better)."""
                # Priority 1: Has entity_id (most important - verified identity)
                has_entity_id = 1 if contact.get('entity_id') else 0

                # Priority 2: Token overlap with cluster mentions
                # Fix 29.5: Include nickname tokens so nickname-indexed contacts rank higher
                contact_tokens = set(contact.get('tokens') or [])
                for nick in (contact.get('nicknames_normalized') or []):
                    if nick:
                        contact_tokens.update(nick.split())
                contact_tokens_phonetic = {normalize_arabic_phonetic(t) for t in contact_tokens}
                overlap = len(contact_tokens_phonetic & mention_tokens_phonetic)

                # Priority 3: Quality tier (HIGH > MED > LOW)
                tier = (contact.get('quality_tier') or '').upper()
                tier_score = {'HIGH': 2, 'MED': 1, 'LOW': 0}.get(tier, 0)

                # Priority 4: Quality score (tiebreaker)
                quality_score = float(contact.get('quality_score') or 0.0)

                return (has_entity_id, overlap, tier_score, quality_score)

            # Sort candidates by priority (descending) and take top MAX
            candidate_contacts.sort(key=candidate_priority, reverse=True)
            candidate_contacts = candidate_contacts[:MAX_GLOBAL_CANDIDATES]

        # Score all candidate contacts
        # Fix 29.13: Track all scored contacts for deduplication-aware margin computation.
        # When the same person exists on multiple phones (e.g., cluster's phone without
        # entity_id + another phone with entity_id), both contacts score identically,
        # producing margin=0 which blocks entity_id propagation. We fix this by:
        # 1. Preferring the contact WITH entity_id when two score equally
        # 2. Computing margin only against contacts with DIFFERENT normalized names
        scored_contacts: List[Tuple[float, bool, Dict, Optional[str]]] = []
        for contact in candidate_contacts:
            score, nickname = self._score_contact(contact, mention_texts)
            has_eid = bool(contact.get('entity_id'))
            scored_contacts.append((score, has_eid, contact, nickname))

        if not scored_contacts:
            return None

        # Sort by score descending, then prefer entity_id (True > False)
        scored_contacts.sort(key=lambda x: (x[0], x[1]), reverse=True)

        best_score_val, _, best_contact, best_nickname = scored_contacts[0]

        if best_contact is None:
            return None

        # Compute margin against the best contact with a DIFFERENT normalized name.
        # This prevents same-person duplicates on different phones from producing margin=0.
        # Fix 29.24: Also strip al-prefix (אל) when comparing names for deduplication.
        # The scoring function does al-prefix stripping, so names like "אלשיח" and "שיח"
        # score identically. Without this fix, Fix 29.13 would see them as "different"
        # names (for margin) but scoring gives them the same score -> margin=0.
        def normalize_for_dedup(name: str) -> str:
            """Normalize name for deduplication: phonetic + al-prefix stripping."""
            norm = normalize_arabic_phonetic(name)
            # Strip אל prefix from each token (matching scoring behavior)
            tokens = norm.split()
            stripped = [AL_PREFIX_PATTERN.sub('', t) for t in tokens]
            return ' '.join(stripped)

        best_name_norm = normalize_for_dedup(best_contact.get('name_normalized', ''))
        second_score = 0.0
        for sc, _, cont, _ in scored_contacts[1:]:
            cont_name_norm = normalize_for_dedup(cont.get('name_normalized', ''))
            if cont_name_norm != best_name_norm:
                second_score = sc
                break

        best_score = best_score_val

        # Fix 25.1 + Fix 29.7: Configurable global match thresholds
        if best_score < self.config.GLOBAL_MATCH_THRESHOLD:
            return None

        # Additional safety: prefer contacts with entity_id
        # If no entity_id, require even higher score
        if not best_contact.get('entity_id') and best_score < self.config.GLOBAL_MATCH_NO_ENTITY_THRESHOLD:
            return None

        margin = best_score - second_score

        # Fix 29.31: Shared Nickname Ambiguity Detection
        # When matched via nickname, check if that nickname is shared by multiple entities.
        # If shared, require disambiguating evidence - otherwise block the match.
        # This prevents incorrect merges when different people share common nicknames.
        if best_nickname:
            # Check if this nickname is shared by multiple entities
            nick_phonetic = normalize_arabic_phonetic(best_nickname)
            nick_entity_ids = self.nickname_entity_ids.get(nick_phonetic, set())
            if not nick_entity_ids:
                # Try original form
                nick_entity_ids = self.nickname_entity_ids.get(best_nickname, set())

            if len(nick_entity_ids) > 1:
                # Nickname is shared! Look for disambiguating evidence.
                has_disambiguating_evidence = False

                # Fix 29.33: Helper to strip al-prefix BEFORE phonetic normalization.
                # Phonetic normalization converts ע→א, making "עלי" (Ali) look like "אלי",
                # which would be incorrectly stripped if we strip after normalization.
                # Solution: strip al-prefix on RAW token first, then normalize.
                def strip_al_raw(tok: str) -> str:
                    """Strip אל- or אל prefix from RAW (non-normalized) token."""
                    if tok.startswith('אל-') and len(tok) > 3:
                        return tok[3:]
                    if tok.startswith('אל') and len(tok) > 2:
                        return tok[2:]
                    return tok

                # Get contact's tokens (given name, family, etc.)
                contact_given = None
                contact_family = None
                contact_family_raw = None  # Keep raw for al-stripping
                contact_tokens_phonetic: Set[str] = set()
                contact_tokens_raw: List[str] = []  # Keep raw for al-stripping
                main_tokens = self.normalizer.tokenize(best_contact.get('name_normalized', ''))
                main_tokens = [t for t in main_tokens if t and t not in self.noise_tokens]
                for t in main_tokens:
                    contact_tokens_phonetic.add(normalize_arabic_phonetic(t))
                    contact_tokens_raw.append(t)
                if main_tokens:
                    first_tok = main_tokens[0]
                    if first_tok != 'אבו' and not first_tok.startswith('אבו-'):
                        contact_given = normalize_arabic_phonetic(first_tok)
                    if len(main_tokens) >= 2:
                        contact_family = normalize_arabic_phonetic(main_tokens[-1])
                        contact_family_raw = main_tokens[-1]  # Keep raw

                # Get cluster's non-kunya tokens (both normalized and raw)
                cluster_non_kunya_tokens: Set[str] = set()
                cluster_non_kunya_tokens_raw: List[str] = []  # Keep raw for al-stripping
                for mention in mention_texts:
                    mention_tokens = self.normalizer.tokenize(mention)
                    mention_tokens = [t for t in mention_tokens if t and t not in self.noise_tokens]
                    for i, tok in enumerate(mention_tokens):
                        # Skip kunya tokens
                        if tok == 'אבו' or tok.startswith('אבו-'):
                            continue
                        # Skip kunya complement (token after standalone אבו)
                        if i > 0 and mention_tokens[i-1] == 'אבו':
                            continue
                        cluster_non_kunya_tokens.add(normalize_arabic_phonetic(tok))
                        cluster_non_kunya_tokens_raw.append(tok)  # Keep raw

                # Evidence 1: Unique nickname in cluster
                # Check if cluster contains a nickname that's UNIQUE to this contact
                contact_nicknames = best_contact.get('nicknames_normalized') or []
                for nick in contact_nicknames:
                    nick_ph = normalize_arabic_phonetic(nick)
                    nick_count = len(self.nickname_entity_ids.get(nick_ph, set()))
                    if nick_count == 0:
                        nick_count = len(self.nickname_entity_ids.get(nick, set()))
                    if nick_count == 1:  # Unique nickname
                        # Check if this unique nickname is contained in cluster
                        nick_tokens = set(normalize_arabic_phonetic(t) for t in self.normalizer.tokenize(nick) if t)
                        all_mention_tokens = set()
                        for m in mention_texts:
                            for t in self.normalizer.tokenize(m):
                                if t:
                                    all_mention_tokens.add(normalize_arabic_phonetic(t))
                        if nick_tokens and nick_tokens.issubset(all_mention_tokens):
                            has_disambiguating_evidence = True
                            break

                # Evidence 2: Given name match
                if not has_disambiguating_evidence and contact_given and cluster_non_kunya_tokens:
                    for cluster_tok in cluster_non_kunya_tokens:
                        if len(cluster_tok) < 2:
                            continue
                        sim = _char_ratio(cluster_tok, contact_given) / 100.0
                        if sim >= 0.90:
                            has_disambiguating_evidence = True
                            break

                # Evidence 3: Family name match
                # Fix 29.33: Strip al-prefix from RAW tokens BEFORE phonetic normalization
                # to avoid false stripping (e.g., "עלי" → "אלי" → incorrectly stripped to "י")
                if not has_disambiguating_evidence and contact_family_raw and cluster_non_kunya_tokens_raw:
                    for cluster_tok_raw in cluster_non_kunya_tokens_raw:
                        if len(cluster_tok_raw) < 2:
                            continue
                        # Strip al-prefix from raw, then normalize for comparison
                        cluster_cmp = normalize_arabic_phonetic(strip_al_raw(cluster_tok_raw))
                        family_cmp = normalize_arabic_phonetic(strip_al_raw(contact_family_raw))
                        if len(cluster_cmp) < 2 or len(family_cmp) < 2:
                            continue
                        sim = _char_ratio(cluster_cmp, family_cmp) / 100.0
                        if sim >= 0.80:
                            has_disambiguating_evidence = True
                            break

                # Evidence 4: Any token overlap with contact's main name tokens
                # Fix 29.33: Strip al-prefix from RAW tokens BEFORE phonetic normalization
                if not has_disambiguating_evidence and cluster_non_kunya_tokens_raw and contact_tokens_raw:
                    for cluster_tok_raw in cluster_non_kunya_tokens_raw:
                        if len(cluster_tok_raw) < 3:
                            continue
                        # Strip al-prefix from raw, then normalize for comparison
                        cluster_cmp = normalize_arabic_phonetic(strip_al_raw(cluster_tok_raw))
                        if len(cluster_cmp) < 2:
                            continue
                        for contact_tok_raw in contact_tokens_raw:
                            if len(contact_tok_raw) < 3:
                                continue
                            contact_cmp = normalize_arabic_phonetic(strip_al_raw(contact_tok_raw))
                            if len(contact_cmp) < 2:
                                continue
                            sim = _char_ratio(cluster_cmp, contact_cmp) / 100.0
                            if sim >= 0.85:
                                has_disambiguating_evidence = True
                                break
                        if has_disambiguating_evidence:
                            break

                if not has_disambiguating_evidence:
                    # No evidence to distinguish this contact from others with same nickname
                    return None

        # Fix 29.20: Cap scores at 1.0 for external API (internal sorting uses uncapped)
        capped_best = min(1.0, best_score)
        capped_second = min(1.0, second_score)
        return Cube2Match(
            name=best_contact['name'],
            nickname=best_nickname,
            score=capped_best,
            second_score=capped_second,
            margin=margin,  # Keep uncapped margin for confident gate
            entity_id=best_contact.get('entity_id'),
            contact_key=best_contact.get('contact_key'),
            name_normalized=best_contact.get('name_normalized', ''),
            tokens=list(best_contact.get('tokens') or []),
            quality_tier=best_contact.get('quality_tier', ''),
            quality_score=float(best_contact.get('quality_score') or 0.0),
            # Fix 44: Store source phone for cross-phone contact lookups
            source_phone=best_contact.get('phone'),
            status=best_contact.get('status'),
            id_number=best_contact.get('id_number'),
        )

    # ------------------------------------------------------------------
    # Internal: contact indexing + quality
    # ------------------------------------------------------------------
    def _index_contacts(self, cube2_df: pd.DataFrame) -> None:
        """Index cube2 contacts by normalized phone number."""
        if cube2_df is None or cube2_df.empty:
            return

        for _, row in cube2_df.iterrows():
            phone = row.get(self.config.COL_CONTACT_PHONE)
            name = row.get(self.config.COL_CONTACT_NAME)
            entity_id = row.get(self.config.COL_CONTACT_ENTITY_ID)
            nickname = row.get(self.config.COL_CONTACT_NICKNAME)
            contact_status = row.get(self.config.COL_CONTACT_STATUS)
            contact_id_number = row.get(self.config.COL_CONTACT_ID_NUMBER)

            if pd.isna(phone) or pd.isna(name):
                continue

            norm_phone = normalize_phone(str(phone))
            if not norm_phone:
                continue

            raw_name = str(name).strip()
            if not raw_name:
                continue

            # Fix 29.14 + Fix 29.26: Extract ALL nicknames from parentheses if verified_nickname is empty
            # Production data often has multiple kunyas: verified_name="פית׳ם שיח׳ צ׳אער (אבו-חמסה) (אבו-אחמד) (אבו-פהד)"
            # Or comma-separated in one: verified_name="פית׳ם שיח׳ צ׳אער (אבו-חמסה, אבו-אחמד)"
            # We need to extract ALL kunyas to enable dual-evidence boosts (Fix 26.8, 27.2)
            # Fix 29.35: Use pd.isna() to properly handle NaN/NA values (np.nan, pd.NA, None)
            # The old check `not nickname` failed for np.nan because bool(np.nan) is True
            nickname_is_empty = pd.isna(nickname) or (isinstance(nickname, str) and not nickname.strip())
            if nickname_is_empty:
                # Fix 29.26: Use findall to get ALL parenthetical content, not just the first
                all_paren_matches = re.findall(r'\(([^)]+)\)', raw_name)
                if all_paren_matches:
                    # Filter for kunyas (start with אבו) - parentheses might contain other data
                    extracted_kunyas = []
                    for paren_content in all_paren_matches:
                        paren_content = paren_content.strip()
                        # Check if this parenthetical contains kunyas
                        # Could be single: "(אבו-חמסה)" or comma-separated: "(אבו-חמסה, אבו-אחמד)"
                        for part in paren_content.replace(',', '،').split('،'):  # Handle both comma types
                            part = part.strip()
                            if part.startswith('אבו') or part.startswith('אבו-'):
                                extracted_kunyas.append(part)

                    if extracted_kunyas:
                        # Join all extracted kunyas with comma - _split_nicknames() will handle it
                        nickname = ', '.join(extracted_kunyas)
                        # Remove ALL parentheticals from main name so they're not scored twice
                        raw_name = re.sub(r'\s*\([^)]+\)\s*', ' ', raw_name).strip()
                        # Collapse multiple spaces
                        raw_name = ' '.join(raw_name.split())

            name_normalized = self.normalizer.normalize(raw_name)
            if not name_normalized:
                continue

            tokens = [t for t in self.normalizer.tokenize(name_normalized) if t and t not in self.noise_tokens]

            entity_id = _coerce_entity_id(entity_id)
            contact_key = self._make_contact_key(entity_id, name_normalized)

            # Normalize and store nicknames (optional)
            nicknames_norm: List[str] = []
            nicknames_raw: List[str] = []  # Parallel list of raw (un-normalized) nicknames
            if isinstance(nickname, str) and nickname.strip():
                for n in _split_nicknames(nickname):
                    n_norm = self.normalizer.normalize(n)
                    if n_norm:
                        nicknames_norm.append(n_norm)
                        nicknames_raw.append(n.strip())

            # Fix 12.1: Best-Field Quality Scoring
            # Calculate quality from BOTH name AND nicknames, use the BEST.
            # This prevents "Mom" with nickname "Rivka Cohen" from being LOW quality.
            name_tier, name_score = self._compute_contact_quality(tokens)
            quality_tier, quality_score = name_tier, name_score
            best_nickname_for_quality: Optional[str] = None
            best_nickname_for_quality_raw: Optional[str] = None

            tier_rank = {'LOW': 0, 'MED': 1, 'HIGH': 2}
            for nick_idx, nick_norm in enumerate(nicknames_norm):
                nick_tokens = [t for t in self.normalizer.tokenize(nick_norm) if t and t not in self.noise_tokens]
                nick_tier, nick_score = self._compute_contact_quality(nick_tokens)
                nick_tier_rank = tier_rank.get(nick_tier, 0)
                curr_tier_rank = tier_rank.get(quality_tier, 0)
                # Fix 21.4: Consider quality_score within same tier, not just tier upgrades
                # Previously, a MED nickname with score=2.5 would not replace MED primary with score=1.5
                if nick_tier_rank > curr_tier_rank or (nick_tier_rank == curr_tier_rank and nick_score > quality_score):
                    quality_tier = nick_tier
                    quality_score = nick_score
                    best_nickname_for_quality = nick_norm
                    best_nickname_for_quality_raw = nicknames_raw[nick_idx] if nick_idx < len(nicknames_raw) else nick_norm

            contact = {
                'phone': norm_phone,
                'name': raw_name,
                'name_normalized': name_normalized,
                'tokens': tokens,
                'entity_id': entity_id,
                'contact_key': contact_key,
                'quality_tier': quality_tier,
                'quality_score': quality_score,
                'nicknames_normalized': nicknames_norm,
                'best_nickname_for_quality': best_nickname_for_quality,  # Track which nickname upgraded quality
                'best_nickname_for_quality_raw': best_nickname_for_quality_raw,  # Raw (un-normalized) version for display
                'status': str(contact_status).strip() if contact_status and not pd.isna(contact_status) else '',
                'id_number': str(contact_id_number).strip() if contact_id_number and not pd.isna(contact_id_number) else '',
            }

            self.contacts_by_phone[norm_phone].append(contact)

            # Fix 25.1: Also index by name for global (cross-phone) lookup
            # This enables matching phonebook entries even when cluster is on different phone
            self.global_name_index[name_normalized].append(contact)

            # Fix 25.2: Also index PHONETIC version to enable cross-transliteration matching
            # This allows cluster "גחאן" (Gimel) to find phonebook "ע׳חאן" (Ghayin)
            # because both normalize to the same phonetic form
            name_phonetic = normalize_arabic_phonetic(name_normalized)
            if name_phonetic and name_phonetic != name_normalized:
                self.global_name_index[name_phonetic].append(contact)

            # Fix 28.3 (Finding 47): Also index HYPHEN-NORMALIZED variant
            # Names like "עבד-אל-דימאן" may be stored in phonebook but searched as
            # "עבד אל דימאן" (with spaces). Index the hyphen-to-space variant.
            name_hyphen_norm = name_normalized.replace('-', ' ')
            if name_hyphen_norm and name_hyphen_norm != name_normalized:
                # Re-normalize after hyphen replacement (handles multi-space collapse)
                name_hyphen_norm = ' '.join(name_hyphen_norm.split())
                self.global_name_index[name_hyphen_norm].append(contact)
                # Also index phonetic variant of hyphen-normalized
                name_hyphen_phonetic = normalize_arabic_phonetic(name_hyphen_norm)
                if name_hyphen_phonetic and name_hyphen_phonetic != name_hyphen_norm:
                    self.global_name_index[name_hyphen_phonetic].append(contact)

            # Fix 26.3: Also index by INDIVIDUAL TOKENS for fuzzy global lookup
            # This enables finding contacts even when full-string lookup fails due to
            # minor spelling differences (e.g., "אלדימאן" vs "אלדימן").
            # The token "מידאוי" (family name) will still match, retrieving the candidate.
            # Only index tokens that are distinctive (not too common/generic).
            for tok in tokens:
                if tok and tok not in self.generic_tokens and len(tok) >= 3:
                    tok_phonetic = normalize_arabic_phonetic(tok)
                    self.global_name_index[tok].append(contact)
                    if tok_phonetic and tok_phonetic != tok:
                        self.global_name_index[tok_phonetic].append(contact)

            # Also index by nicknames for global lookup
            for nick_norm in nicknames_norm:
                if nick_norm:
                    self.global_name_index[nick_norm].append(contact)
                    # Also index phonetic version of nicknames
                    nick_phonetic = normalize_arabic_phonetic(nick_norm)
                    if nick_phonetic and nick_phonetic != nick_norm:
                        self.global_name_index[nick_phonetic].append(contact)

                    # Fix 29.28: Track nickname frequency for unique nickname bonus
                    # Use contact_key as a stable identifier (entity_id when present, else name hash)
                    # to count unique entities rather than duplicate contacts
                    self.nickname_entity_ids[nick_norm].add(contact_key)
                    if nick_phonetic and nick_phonetic != nick_norm:
                        self.nickname_entity_ids[nick_phonetic].add(contact_key)

    def _make_contact_key(self, entity_id: Optional[str], name_normalized: str) -> str:
        if entity_id:
            return f'EID:{entity_id}'
        # Stable within-run + cross-run (deterministic) name key
        h = hashlib.sha1(name_normalized.encode('utf-8')).hexdigest()[:12]
        return f'NAME:{h}'

    def _compute_contact_quality(self, tokens: List[str]) -> Tuple[str, float]:
        """Conservative phonebook-contact quality tiering (LOW/MED/HIGH).

        Fix 24.3: Handle "Kunya Trap" - hyphenated kunya prefixes (אבו-X, אבן-X, אם-X)
        are treated as 1 token by split(), but semantically represent 2 name parts.
        Without this fix, "אבו-חמסה" (Abu-Hamsa) becomes 1 token and fails the
        CUBE2_PHONEBOOK_MIN_TOKENS=2 check, causing LOW quality classification.
        """
        # Fix 24.3: Calculate effective token count
        # Hyphenated kunya prefixes count as 2 tokens (prefix + name)
        # Only אבו- (Abu - "father of") is common enough to warrant this treatment
        KUNYA_PREFIXES = ('אבו-',)

        def _effective_count(token: str) -> int:
            """Return 2 for kunya-prefixed tokens, 1 otherwise."""
            if any(token.startswith(prefix) for prefix in KUNYA_PREFIXES):
                return 2
            return 1

        effective_token_count = sum(_effective_count(t) for t in tokens)

        # Require at least N tokens to be considered name-like
        if effective_token_count < int(getattr(self.config, 'CUBE2_PHONEBOOK_MIN_TOKENS', 2)):
            return 'LOW', float(effective_token_count)

        # Remove generic/role tokens (in normalized space)
        # Note: Kunya prefixes themselves are NOT generic - they're meaningful identifiers
        non_generic = [t for t in tokens if t not in self.generic_tokens]

        # Fix 24.3: Also calculate effective non-generic count for kunya tokens
        effective_non_generic_count = sum(_effective_count(t) for t in non_generic)

        min_non_generic = int(getattr(self.config, 'CUBE2_PHONEBOOK_MIN_NON_GENERIC_TOKENS', 2))

        if effective_non_generic_count >= min_non_generic:
            # HIGH: all tokens look person-like (no generic tokens)
            if effective_non_generic_count == effective_token_count:
                tier = 'HIGH'
            else:
                tier = 'MED'
        else:
            tier = 'LOW'

        # Simple numeric score (for observability, not a strict model feature)
        # Use effective counts to give proper weight to kunya names
        effective_generic_count = effective_token_count - effective_non_generic_count
        score = float(effective_non_generic_count) - 0.10 * float(effective_generic_count)
        return tier, max(score, 0.0)

    def _score_contact(self, contact: Dict[str, Any], mention_texts: List[str]) -> Tuple[float, Optional[str]]:
        """Score a contact against a list of mention texts.

        Returns (best_score, best_matching_nickname_or_None).

        Fix 26.1: Nickname Hijack Protection - When the best match is via nickname,
        validates that the phonebook's main name is consistent with the cluster's
        full-name mentions. Rejects matches where family names clearly conflict.

        Fix 26.8: Dual-Evidence Boost - When both main name AND nickname match
        reasonably well, boost the score because we have corroborating evidence.
        This helps pass the 0.95 threshold for global matching without entity_id.
        """
        contact_main = contact.get('name_normalized', '')
        nicknames = list(contact.get('nicknames_normalized') or [])

        # Track main name score and best nickname score separately
        main_score = 0.0
        for mention_text in mention_texts:
            if contact_main:
                score = self._text_similarity(mention_text, contact_main)
                main_score = max(main_score, score)

        best_nick_score = 0.0
        best_nickname = None
        # Fix 29.28: Track all matching nicknames to detect unique ones
        matching_nicknames: List[Tuple[float, str]] = []
        for mention_text in mention_texts:
            for nick in nicknames:
                score = self._text_similarity(mention_text, nick)
                if score > best_nick_score:
                    best_nick_score = score
                    best_nickname = nick
                # Track all decent matches for unique nickname detection
                if score >= 0.80:
                    matching_nicknames.append((score, nick))

        # Determine best overall score and whether it came from nickname
        if best_nick_score > main_score:
            best = best_nick_score
            matched_nickname = best_nickname
        else:
            best = main_score
            matched_nickname = None

        # Fix 29.28: Unique Nickname Scoring Bonus
        # When the cluster CONTAINS a UNIQUE nickname (only this contact has it),
        # add a small bonus to create margin against contacts with shared nicknames.
        # Example: E1 has ["אבו-חמסה", "אבו-אחמד"], E2 has ["אבו-אחמד", "אבו-מחמד"]
        #   - Cluster mentions include "אבו-חמסה אבו-אחמד פית׳ם" which contains "אבו-חמסה"
        #   - "אבו-חמסה" is unique to E1 → E1 gets bonus, E2 doesn't → margin
        # Uses containment check instead of fuzzy similarity to avoid false matches
        # between similar kunyas like "אבו-אחמד" vs "אבו-מחמד"
        UNIQUE_NICK_BONUS = 0.05  # Small bonus to create margin
        has_unique_nickname_match = False

        # Build token sets from all mentions for containment check
        all_mention_tokens: Set[str] = set()
        for mention_text in mention_texts:
            for tok in self.normalizer.tokenize(mention_text):
                if tok and tok not in self.noise_tokens:
                    all_mention_tokens.add(normalize_arabic_phonetic(tok))

        # Check each nickname for uniqueness and containment
        for nick in nicknames:
            nick_phonetic = normalize_arabic_phonetic(nick)
            entity_count = len(self.nickname_entity_ids.get(nick_phonetic, set()))
            if entity_count == 0:
                # Nickname not in phonetic index, check original
                entity_count = len(self.nickname_entity_ids.get(nick, set()))

            if entity_count != 1:
                continue  # Not unique, skip

            # Check if this unique nickname is contained in any mention
            nick_tokens = set(normalize_arabic_phonetic(t) for t in self.normalizer.tokenize(nick) if t)
            if nick_tokens and nick_tokens.issubset(all_mention_tokens):
                has_unique_nickname_match = True
                break  # Found a unique match, apply bonus

        if has_unique_nickname_match:
            best = best + UNIQUE_NICK_BONUS

        # Fix 26.8: Dual-Evidence Boost
        # When BOTH main name AND nickname match well, boost confidence.
        # This provides corroborating evidence that helps pass 0.95 threshold.
        # Example: Cluster "אבו-בעאד׳ אלואליה פגיד אלואליה"
        #          Contact name "פגיד אלואליה" + nickname "אבו-בעאד׳"
        #          Main match ~91.6%, Nick match ~86% → boosted to ~96.6%
        DUAL_EVIDENCE_MAIN_THRESHOLD = 0.85
        DUAL_EVIDENCE_NICK_THRESHOLD = 0.80
        DUAL_EVIDENCE_BOOST = 0.05

        if main_score >= DUAL_EVIDENCE_MAIN_THRESHOLD and best_nick_score >= DUAL_EVIDENCE_NICK_THRESHOLD:
            # Fix 29.20: Don't cap at 1.0 here - allow dual-evidence matches to rank
            # higher than single-evidence matches for sorting/margin purposes.
            # The cap is applied at the end of match_global when building Cube2Match.
            best = best + DUAL_EVIDENCE_BOOST

        # Fix 27.2: Nickname-Family Corroboration Boost
        # When the best match is via nickname AND the cluster's family token matches
        # the phonebook main's family token, this is strong corroborating evidence.
        # This handles cases where nickname matches perfectly but main_score is low
        # because the phonebook has extra tokens (middle names) that cluster lacks.
        # Example: Cluster "אבו-פלס חלב חג׳אג׳" vs Phonebook "חלב מחמוד לג׳ס חג׳אג׳ (אבו-פלס)"
        #   - nick_score = 0.88 (perfect subset, but length penalty)
        #   - main_score = 0.60 (extra tokens מחמוד לג׳ס)
        #   - Family "חג׳אג׳" matches in both → boost to 0.93
        NICK_FAMILY_BOOST_THRESHOLD = 0.85
        NICK_FAMILY_BOOST = 0.05
        FAMILY_MATCH_MIN_SIM = 0.80
        NICK_FAMILY_BOOST_MAIN_FLOOR = 0.45  # Require SOME meaningful main name overlap

        # The main_score floor (0.45) prevents false positives when:
        # - Two people share the same kunya (e.g., both named son "חסן")
        # - Two people share a common family name (e.g., "כהן")
        # - But given names are completely different (main_score ~0.30)
        # In the target case (אבו-פלס חלב חג׳אג׳ vs חלב מחמוד לג׳ס חג׳אג׳), main_score ~0.60
        # because both "חלב" and "חג׳אג׳" are shared, indicating real identity overlap.
        if (matched_nickname and
            best_nick_score >= NICK_FAMILY_BOOST_THRESHOLD and
            main_score >= NICK_FAMILY_BOOST_MAIN_FLOOR and
            main_score < DUAL_EVIDENCE_MAIN_THRESHOLD):
            # Fix 29.33: Helper to strip al-prefix BEFORE phonetic normalization.
            # Phonetic normalization converts ע→א, making "עלי" (Ali) look like "אלי",
            # which would be incorrectly stripped if we strip after normalization.
            def _strip_al_raw(text: str) -> str:
                """Strip אל- or אל prefix from RAW (non-normalized) token."""
                if text.startswith('אל-') and len(text) > 3:
                    return text[3:]
                if text.startswith('אל') and len(text) > 2:
                    return text[2:]
                return text

            # Extract family token from cluster (last non-kunya token from longest mention)
            # Fix 29.33: Keep RAW token for al-stripping before normalization
            cluster_family_raw = None
            for mention in sorted(mention_texts, key=len, reverse=True):
                mention_tokens = self.normalizer.tokenize(mention)
                mention_tokens = [t for t in mention_tokens if t not in self.noise_tokens]
                if len(mention_tokens) >= 2:
                    first_tok = mention_tokens[0]
                    # Check for kunya pattern (אבו X or אבו-X)
                    if first_tok == 'אבו' or first_tok.startswith('אבו-'):
                        # Fix 29.27: Multi-Kunya Family Extraction
                        # When multiple tokens are kunyas, remaining tokens may be just given names
                        # Example: "אבו-חמסה אבו-אחמד פית׳ם" = 2 kunyas + 1 given name, NO family
                        kunya_count = sum(
                            1 for t in mention_tokens
                            if t == 'אבו' or t.startswith('אבו-')
                        )
                        non_kunya_count = len(mention_tokens) - kunya_count
                        # Fix 29.34: Only skip when MULTIPLE kunyas exist AND only 1 non-kunya token.
                        # Single-kunya patterns like "אבו-אחמד עלי" (kunya + family) should extract family.
                        # Multi-kunya patterns like "אבו-חמסה אבו-אחמד פית׳ם" with only 1 non-kunya
                        # should skip because that token is likely a given name, not family.
                        if kunya_count >= 2 and non_kunya_count < 2:
                            continue  # Multiple kunyas with only 1 non-kunya = no family

                        # Fix 29.25: Allow 2-token kunya-start patterns
                        # Previously required >= 3 tokens (kunya + given + family),
                        # but "אבו-אחמד כהן" is a valid 2-token pattern (kunya + family).
                        # We extract token[-1] and let the family match check determine
                        # if it's actually a family name. This is safe because we're not
                        # asserting it IS a family, just checking IF it matches the
                        # phonebook family - mismatches won't get the boost.
                        if len(mention_tokens) >= 2:
                            cluster_family_raw = mention_tokens[-1]  # Keep raw
                            break
                    else:
                        # Fix 27.3: Handle kunya-at-END pattern
                        # Case 1: Hyphenated kunya "אבו-אנס" stays as single token
                        #         "אחמד אלברביר אבו-אנס" → ['אחמד', 'אלברביר', 'אבו-אנס']
                        #         Family is at position -2
                        # Case 2: Unhyphenated kunya splits into two tokens
                        #         "אחמד אלברביר אבו אנס" → ['אחמד', 'אלברביר', 'אבו', 'אנס']
                        #         Family is at position -3
                        if mention_tokens[-1].startswith('אבו-'):
                            # Case 1: Hyphenated kunya at end
                            if len(mention_tokens) >= 3:
                                cluster_family_raw = mention_tokens[-2]  # Keep raw
                                break
                            else:
                                # Too short, try next mention
                                continue
                        elif len(mention_tokens) >= 4 and mention_tokens[-2] == 'אבו':
                            # Case 2: Unhyphenated kunya at end
                            cluster_family_raw = mention_tokens[-3]  # Keep raw
                            break
                        elif len(mention_tokens) >= 2 and mention_tokens[-2] == 'אבו':
                            # Short mention with kunya at end - no family, try next mention
                            continue
                        else:
                            cluster_family_raw = mention_tokens[-1]  # Keep raw
                            break

            # Extract family token from contact main (last token)
            # Fix 29.33: Keep RAW token for al-stripping before normalization
            main_family_raw = None
            if contact_main:
                main_tokens = self.normalizer.tokenize(contact_main)
                main_tokens = [t for t in main_tokens if t not in self.noise_tokens]
                if len(main_tokens) >= 2:
                    # Skip if contact main is a kunya name
                    first_main_tok = main_tokens[0]
                    if first_main_tok != 'אבו' and not first_main_tok.startswith('אבו-'):
                        main_family_raw = main_tokens[-1]  # Keep raw

            # Check family match and apply boost
            # Fix 29.33: Strip al-prefix from RAW tokens BEFORE normalizing to avoid
            # false stripping (e.g., "עלי" → "אלי" → incorrectly stripped to "י")
            if cluster_family_raw and main_family_raw and len(cluster_family_raw) >= 2 and len(main_family_raw) >= 2:
                # Strip al-prefix from raw, then normalize for comparison
                cluster_family_cmp = normalize_arabic_phonetic(_strip_al_raw(cluster_family_raw))
                main_family_cmp = normalize_arabic_phonetic(_strip_al_raw(main_family_raw))

                if len(cluster_family_cmp) >= 2 and len(main_family_cmp) >= 2:
                    family_sim = _char_ratio(cluster_family_cmp, main_family_cmp) / 100.0

                    if family_sim >= FAMILY_MATCH_MIN_SIM:
                        # Fix 29.20: Don't cap at 1.0 - allow corroborated matches to
                        # rank higher for sorting/margin. Cap applied in match_global.
                        best = best + NICK_FAMILY_BOOST

        # Fix 29.30: Given Name Corroboration Boost
        # When matched via nickname, if the cluster has a non-kunya token that matches
        # the contact's given name (first non-kunya token), this is corroborating evidence.
        # This helps disambiguate when two contacts share the same nickname but only one
        # has a matching given name.
        # Example: Cluster "אבו-אחמד אחמד", E1 = "מחמד אל-מסרי", E2 = "אחמד אל-ג׳אברי"
        #   - Both share nickname "אבו-אחמד"
        #   - "אחמד" in cluster matches E2's given name → E2 gets boost
        GIVEN_NAME_BOOST = 0.03  # Smaller than family boost - given names more common
        GIVEN_NAME_MIN_SIM = 0.90  # Stricter threshold for given name matching

        if matched_nickname and best_nick_score >= 0.80:
            # Extract non-kunya tokens from cluster mentions
            cluster_non_kunya_tokens: Set[str] = set()
            for mention in mention_texts:
                mention_tokens = self.normalizer.tokenize(mention)
                mention_tokens = [t for t in mention_tokens if t not in self.noise_tokens]
                for tok in mention_tokens:
                    # Skip kunya tokens (standalone אבו or hyphenated אבו-X)
                    if tok == 'אבו' or tok.startswith('אבו-'):
                        continue
                    # Also skip if this is the kunya complement in "אבו X" pattern
                    tok_idx = mention_tokens.index(tok)
                    if tok_idx > 0 and mention_tokens[tok_idx - 1] == 'אבו':
                        continue
                    cluster_non_kunya_tokens.add(normalize_arabic_phonetic(tok))

            # Extract given name from contact (first non-kunya token)
            contact_given = None
            if contact_main:
                main_tokens = self.normalizer.tokenize(contact_main)
                main_tokens = [t for t in main_tokens if t not in self.noise_tokens]
                if len(main_tokens) >= 1:
                    first_tok = main_tokens[0]
                    if first_tok != 'אבו' and not first_tok.startswith('אבו-'):
                        contact_given = normalize_arabic_phonetic(first_tok)

            # Check if any cluster non-kunya token matches the contact's given name
            if contact_given and cluster_non_kunya_tokens:
                for cluster_tok in cluster_non_kunya_tokens:
                    if len(cluster_tok) < 2 or len(contact_given) < 2:
                        continue
                    given_sim = _char_ratio(cluster_tok, contact_given) / 100.0

                    if given_sim >= GIVEN_NAME_MIN_SIM:
                        best = best + GIVEN_NAME_BOOST
                        break  # Only apply boost once

        # Fix 29.36: Kunya Complement Corroboration Boost
        # When a kunya like "אבו-אנס" (Father of Anas) appears, and the contact's main name
        # contains the complement "אנס" (Anas), this strongly corroborates the match.
        # This applies in TWO scenarios:
        # 1. When matched via nickname (matched_nickname is truthy)
        # 2. When cluster mentions contain a kunya and contact name has the complement
        # Example: Cluster "אחמד אלברביר אבו-אנס", Contact1 "אחמד אנס חוסין ברביר (אבו-אנס)"
        #          vs Contact2 "אחמד חוסין אלברביר (אבו-אנס)"
        #   - Both share nickname "אבו-אנס" (Father of Anas)
        #   - Contact1 has "אנס" in name → confirms kunya relationship → gets boost
        #   - Contact2 has no "אנס" → weaker claim to kunya
        KUNYA_COMPLEMENT_BOOST = 0.10  # Significant boost - confirms kunya relationship, creates margin
        KUNYA_COMPLEMENT_MIN_SIM = 0.85  # Allow minor spelling variations

        def _extract_kunya_complement(kunya_text: str) -> Optional[str]:
            """Extract the complement from a kunya (e.g., "אבו-אנס" → "אנס")."""
            if not kunya_text:
                return None
            kunya_norm = normalize_arabic_phonetic(kunya_text.strip())
            if kunya_norm.startswith('אבו-'):
                return kunya_norm[4:]  # Skip "אבו-"
            elif kunya_norm.startswith('אבו '):
                parts = kunya_norm.split()
                if len(parts) >= 2:
                    return parts[1]
            return None

        def _contact_has_complement(complement: str, contact_main_text: str) -> bool:
            """Check if contact's main name contains the kunya complement as a MIDDLE name.

            In Arabic naming, "אבו-X" (Father of X) means the person has a son named X.
            The son's name appears as a MIDDLE NAME in the father's full name, not as
            the given name (first position). For example:
              - "אחמד אנס חוסין ברביר" - "אנס" at position 1 is the son's name
              - "אחמד אל-ג׳אברי" - "אחמד" at position 0 is HIS name, not his son's

            We skip position 0 (given name) to avoid false positives where the contact's
            own name matches the kunya complement.
            """
            if not complement or len(complement) < 2 or not contact_main_text:
                return False
            main_tokens = self.normalizer.tokenize(contact_main_text)
            main_tokens_normalized = [normalize_arabic_phonetic(t) for t in main_tokens
                                      if t not in self.noise_tokens]
            # Skip the first token (given name) - complement should be a middle name
            if len(main_tokens_normalized) < 2:
                return False  # Need at least 2 tokens (given + middle/family)
            for main_tok in main_tokens_normalized[1:]:  # Start from position 1
                if len(main_tok) < 2:
                    continue
                comp_sim = _char_ratio(complement, main_tok) / 100.0
                if comp_sim >= KUNYA_COMPLEMENT_MIN_SIM:
                    return True
            return False

        kunya_boost_applied = False

        # Safety check: Extract families from cluster and contact to ensure no conflict
        # The kunya complement boost should NOT override strong family evidence
        def _extract_cluster_families() -> Set[str]:
            """Extract family tokens from cluster mentions."""
            families: Set[str] = set()
            for mention in mention_texts:
                mention_tokens = self.normalizer.tokenize(mention)
                mention_tokens = [t for t in mention_tokens if t not in self.noise_tokens]
                if len(mention_tokens) < 2:
                    continue
                # Skip kunya-only mentions based on token count
                first_tok = mention_tokens[0]
                if first_tok == 'אבו':
                    # Unhyphenated kunya: needs "אבו X family" = 3+ tokens
                    if len(mention_tokens) < 3:
                        continue
                elif first_tok.startswith('אבו-'):
                    # Hyphenated kunya: "אבו-X family" = 2 tokens OK
                    # But "אבו-X" alone = 1 token, skip
                    if len(mention_tokens) < 2:
                        continue
                # Get family (last non-kunya token)
                family_tok = mention_tokens[-1]
                if family_tok.startswith('אבו-') or family_tok == 'אבו':
                    if len(mention_tokens) >= 2:
                        family_tok = mention_tokens[-2]
                if family_tok and family_tok != 'אבו' and not family_tok.startswith('אבו-'):
                    # Strip al-prefix before normalizing
                    if family_tok.startswith('אל-') and len(family_tok) > 3:
                        family_tok = family_tok[3:]
                    elif family_tok.startswith('אל') and len(family_tok) > 2:
                        family_tok = family_tok[2:]
                    families.add(normalize_arabic_phonetic(family_tok))
            return families

        def _extract_contact_family() -> Optional[str]:
            """Extract family token from contact main name."""
            if not contact_main:
                return None
            main_tokens = self.normalizer.tokenize(contact_main)
            main_tokens = [t for t in main_tokens if t not in self.noise_tokens]
            if len(main_tokens) < 2:
                return None
            family_tok = main_tokens[-1]
            # Strip al-prefix before normalizing
            if family_tok.startswith('אל-') and len(family_tok) > 3:
                family_tok = family_tok[3:]
            elif family_tok.startswith('אל') and len(family_tok) > 2:
                family_tok = family_tok[2:]
            return normalize_arabic_phonetic(family_tok) if family_tok else None

        def _families_compatible(cluster_families: Set[str], contact_family: Optional[str]) -> bool:
            """Check if contact family is compatible with cluster families."""
            if not cluster_families:
                return True  # No cluster family evidence - no conflict possible
            if not contact_family:
                return True  # No contact family - no conflict possible
            # Check if contact family matches ANY cluster family (with fuzzy matching)
            for cf in cluster_families:
                if len(cf) < 2 or len(contact_family) < 2:
                    continue
                sim = _char_ratio(cf, contact_family) / 100.0
                if sim >= 0.80:  # 80% match threshold
                    return True
            return False  # Contact family conflicts with cluster families

        # Pre-compute family compatibility for kunya boost safety check
        cluster_families = _extract_cluster_families()
        contact_family = _extract_contact_family()
        families_ok = _families_compatible(cluster_families, contact_family)

        # Scenario 1: Matched via nickname - check if nickname is a kunya
        if matched_nickname and best_nick_score >= 0.80:
            kunya_complement = _extract_kunya_complement(matched_nickname)
            if kunya_complement and _contact_has_complement(kunya_complement, contact_main):
                # Only apply boost if families don't conflict
                if families_ok:
                    best = best + KUNYA_COMPLEMENT_BOOST
                    kunya_boost_applied = True

        # Scenario 2: Cluster mentions contain a kunya - check if contact has complement
        # This applies even when matched via main name (matched_nickname is None)
        if not kunya_boost_applied and contact_main:
            # Extract kunyas from cluster mentions
            cluster_kunyas: Set[str] = set()
            for mention in mention_texts:
                mention_tokens = self.normalizer.tokenize(mention)
                for i, tok in enumerate(mention_tokens):
                    if tok.startswith('אבו-'):
                        # Hyphenated kunya
                        cluster_kunyas.add(tok)
                    elif tok == 'אבו' and i + 1 < len(mention_tokens):
                        # Spaced kunya: "אבו" + next token
                        cluster_kunyas.add(f'אבו-{mention_tokens[i+1]}')

            # Check if any cluster kunya's complement appears in contact name
            for kunya in cluster_kunyas:
                kunya_complement = _extract_kunya_complement(kunya)
                if kunya_complement and _contact_has_complement(kunya_complement, contact_main):
                    # Only apply boost if families don't conflict
                    if families_ok:
                        best = best + KUNYA_COMPLEMENT_BOOST
                    break  # Only apply boost once

        # Fix 26.1: Nickname Hijack Protection
        # If match is via nickname, verify phonebook main name doesn't conflict with cluster's full names
        if matched_nickname and best >= 0.75:
            if self._detect_nickname_hijack(contact, mention_texts):
                return 0.0, None  # Reject match due to family name conflict

        return best, matched_nickname

    def _detect_nickname_hijack(self, contact: Dict[str, Any], mention_texts: List[str]) -> bool:
        """Detect if a nickname match would hijack the cluster identity (Fix 26.1).

        Returns True if:
        - Cluster has full-name mentions (2+ tokens, NOT kunya)
        - Phonebook's main name has a family token that CONFLICTS with cluster's family tokens

        This prevents cases like:
        - Cluster: "אבו-חמסה", "פית׳ם שיח׳ צ׳אער"
        - Phonebook: "פית׳ם שיח׳ ח׳ליל (אבו-חמסה)"
        - Match via "אבו-חמסה" would wrongly adopt "ח׳ליל" instead of "צ׳אער"
        """
        contact_main = contact.get('name_normalized', '')
        if not contact_main:
            return False

        # Kunya prefix - only אבו (father of) is treated as kunya in this algorithm
        KUNYA_PREFIXES = {'אבו'}

        # Extract family name from contact (last meaningful token)
        contact_tokens = self.normalizer.tokenize(contact_main)
        contact_tokens = [t for t in contact_tokens if t not in self.noise_tokens]
        if len(contact_tokens) < 2:
            return False  # Contact main name too short to determine family

        # Skip if contact is a kunya name (no family to compare)
        first_token = contact_tokens[0]
        if first_token in KUNYA_PREFIXES:
            return False

        contact_family = normalize_arabic_phonetic(contact_tokens[-1])
        if not contact_family or len(contact_family) < 2:
            return False

        # Find cluster's full-name mentions and extract their family tokens
        # Skip kunya names - they don't have family names in the traditional sense
        cluster_families = set()
        for mention in mention_texts:
            mention_tokens = self.normalizer.tokenize(mention)
            mention_tokens = [t for t in mention_tokens if t not in self.noise_tokens]
            if len(mention_tokens) < 2:
                continue  # Not a full name, skip

            # Skip kunya names - "אבו אחמד" has child's name, not family name
            first_mention_token = mention_tokens[0]
            if first_mention_token in KUNYA_PREFIXES:
                continue

            # Fix 27.3 addendum: Skip short hyphenated kunya at start
            # "אבו-אנס אחמד" (2 tokens) = kunya + given name, NO family
            # But "אבו-אנס אחמד אלברביר" (3+ tokens) = kunya + given + family, OK
            # Fix 29.37: Exception - "אבו-X אל-Y" (2 tokens) = kunya + family name, HAS family
            # Family names in Arabic often start with "אל" (the article)
            # Fix 29.37b: Also extract family if second token is NOT a common given name
            # Example: "אבו-אנס ברביר" → "ברביר" is NOT a given name → extract as family
            if first_mention_token.startswith('אבו-') and len(mention_tokens) < 3:
                if len(mention_tokens) == 2:
                    second_token = mention_tokens[1]
                    second_token_norm = normalize_arabic_phonetic(second_token.lstrip('אל').lstrip('-'))
                    # Check if second token looks like a family name:
                    # 1. Starts with "אל" (the article) - clearly a family name
                    # 2. NOT a common given name - likely a family name
                    is_al_family = second_token.startswith('אל')
                    # Fix 29.38: Use normalized set for comparison with normalized token
                    is_common_given = second_token_norm in self.config.get_common_given_names_normalized()
                    if is_al_family or not is_common_given:
                        pass  # Don't skip - second token is likely a family name
                    else:
                        continue  # Second token is a common given name, not a family
                else:
                    continue

            # Fix 29.27: Multi-Kunya Family Extraction
            # When multiple tokens are kunyas, the remaining tokens may be just given names
            # Example: "אבו-חמסה אבו-אחמד פית׳ם" = 2 kunyas + 1 given name, NO family
            # Count kunya tokens (standalone 'אבו' or hyphenated 'אבו-X')
            kunya_count = sum(
                1 for t in mention_tokens
                if t == 'אבו' or t.startswith('אבו-')
            )
            # If kunyas take up all but one token, that remaining token is likely a given name
            # (e.g., 2 kunyas + 1 given = 3 tokens, kunya_count=2, 3-2=1 non-kunya)
            # Need at least 2 non-kunya tokens for family extraction (given + family)
            # Fix 29.37: Exception - if remaining token starts with "אל", it's likely a family name
            # Example: "אבו-אנס אל-עלי" (1 kunya + 1 family) should extract "אל-עלי" as family
            # Fix 29.37b: Also allow if remaining token is NOT a common given name
            # Example: "אבו-אנס ברביר" (1 kunya + 1 family) should extract "ברביר" as family
            if kunya_count >= len(mention_tokens) - 1:
                # Check if there's a family-looking token:
                # 1. Starts with "אל" (the article) - clearly a family name
                # 2. NOT a common given name - likely a family name
                non_kunya_tokens = [t for t in mention_tokens if not (t == 'אבו' or t.startswith('אבו-'))]
                has_family_token = False
                for tok in non_kunya_tokens:
                    tok_norm = normalize_arabic_phonetic(tok.lstrip('אל').lstrip('-'))
                    is_al_family = tok.startswith('אל')
                    # Fix 29.38: Use normalized set for comparison with normalized token
                    is_common_given = tok_norm in self.config.get_common_given_names_normalized()
                    if is_al_family or not is_common_given:
                        has_family_token = True
                        break
                if not has_family_token:
                    continue  # Not enough non-kunya tokens for a family name

            # Fix 27.3: Handle kunya-at-END pattern
            # Case 1: Hyphenated kunya as single token (e.g., "אחמד אלברביר אבו-אנס")
            #         Tokenizes to ['אחמד', 'אלברביר', 'אבו-אנס']
            #         Family is at position -2 (before the hyphenated kunya)
            # Case 2: Unhyphenated kunya as two tokens (e.g., "אחמד אלברביר אבו אנס")
            #         Tokenizes to ['אחמד', 'אלברביר', 'אבו', 'אנס']
            #         Family is at position -3 (before the two kunya tokens)
            if mention_tokens[-1].startswith('אבו-'):
                # Case 1: Hyphenated kunya at end (single token)
                if len(mention_tokens) >= 3:
                    mention_family = normalize_arabic_phonetic(mention_tokens[-2])
                else:
                    # Too short: "אחמד אבו-אנס" → only first name + kunya, no family
                    continue
            elif len(mention_tokens) >= 4 and mention_tokens[-2] in KUNYA_PREFIXES:
                # Case 2: Unhyphenated kunya at end (two tokens)
                mention_family = normalize_arabic_phonetic(mention_tokens[-3])
            elif len(mention_tokens) >= 2 and mention_tokens[-2] in KUNYA_PREFIXES:
                # Short mention like "אחמד אבו אנס" - no family available, skip
                continue
            else:
                # Normal case - family is last token
                mention_family = normalize_arabic_phonetic(mention_tokens[-1])

            if mention_family and len(mention_family) >= 2:
                cluster_families.add(mention_family)

        if not cluster_families:
            return False  # No full names (non-kunya) in cluster, no conflict possible

        # Check if contact's family matches ANY of the cluster's families
        # Fix 26.5: Strip Al-prefix (אל) before comparison
        # "אלברביר" (Al-Barbir) should match "ברביר" (Barbir)
        def strip_al_prefix(text: str) -> str:
            if text.startswith('אל') and len(text) > 2:
                return text[2:]
            return text

        contact_family_cmp = strip_al_prefix(contact_family)

        for cluster_family in cluster_families:
            cluster_family_cmp = strip_al_prefix(cluster_family)

            family_sim = _char_ratio(contact_family_cmp, cluster_family_cmp) / 100.0

            if family_sim >= 0.70:
                return False  # Found a matching family, no hijack

        # Fix 29.29: Given Name Corroboration
        # Before declaring hijack, check if the cluster's "extracted family" actually
        # matches ANY token in the contact (not just the family). This handles cases like:
        #   Mention: "שיח׳ מחמד" → extracted family = "מחמד"
        #   Contact: "מחמד אל-מסרי" → family = "מסרי", given = "מחמד"
        # The extracted "מחמד" matches the contact's GIVEN name, which is corroborating
        # evidence, not a conflict. This happens when:
        #   - First token is a non-kunya nickname (like "שיח׳")
        #   - Second token is the given name (not family)
        # Only apply this check for short mentions (2 tokens) where the "family extraction"
        # is most likely to be wrong (actually extracting given name as family).
        contact_tokens_normalized = [
            strip_al_prefix(normalize_arabic_phonetic(t))
            for t in contact_tokens if t not in self.noise_tokens
        ]

        for cluster_family in cluster_families:
            cluster_family_cmp = strip_al_prefix(cluster_family)
            # Check against ALL contact tokens (given name, middle names, family)
            for contact_token in contact_tokens_normalized:
                token_sim = _char_ratio(contact_token, cluster_family_cmp) / 100.0

                if token_sim >= 0.85:  # Stricter threshold for non-family match
                    return False  # Cluster "family" matches a contact token, not a hijack

        # Contact's family doesn't match any cluster family - this is a hijack!
        return True

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity.

        Fix 24.2: Containment Boost - When one name is a subset of another
        (e.g., "נסיר ג׳אהר" vs "אבו-פאחר נסיר ג׳אהר"), the token_set_ratio
        correctly identifies 100% match but fuzz.ratio penalizes the length
        difference. This drags the score below threshold.

        Solution: When token_set_ratio >= 0.90 (high-confidence subset match),
        increase its weight to 0.8 to override the length penalty from fuzz.ratio.
        """
        if not text1 or not text2:
            return 0.0

        # Fix 13.1: Apply Arabic phonetic normalization for consistency with SimilarityScorer
        # This ensures "Al-Mahmoud" matches "Mahmoud" and handles ק/כ, ט/ת, ע/א homophones
        text1_norm = normalize_arabic_phonetic(text1)
        text2_norm = normalize_arabic_phonetic(text2)

        # Also try stripping "אל" prefix for Arabic definite article matching
        text1_stripped = AL_PREFIX_PATTERN.sub('', text1_norm).strip() if text1_norm else ''
        text2_stripped = AL_PREFIX_PATTERN.sub('', text2_norm).strip() if text2_norm else ''

        # Compare both with and without prefix, take best score
        ratio1 = _char_ratio(text1_norm, text2_norm) / 100.0
        token_ratio1 = _token_set_ratio(text1_norm, text2_norm) / 100.0

        # Fix 26.6: Kunya-Aware Scoring Boost
        # When token_set_ratio is high but fuzz.ratio is low due to kunya prefix,
        # strip the kunya and recalculate ratio for a fairer comparison.
        # Example: "אבו-פאחר נסיר ג׳אהר" vs "נסיר ג׳אהר"
        #   - token_set_ratio = 100% (perfect subset)
        #   - fuzz.ratio = ~65% (penalized by "אבו-פאחר" prefix)
        #   - After stripping: ratio("נסיר ג׳אהר", "נסיר ג׳אהר") = 100%
        if token_ratio1 >= 0.90 and ratio1 < 0.85:
            def _strip_kunya_prefix(text: str) -> str:
                """Strip kunya prefix (אבו-X or אבו X) from the beginning."""
                tokens = text.split()
                if tokens and (tokens[0].startswith('אבו-') or tokens[0] == 'אבו'):
                    return ' '.join(tokens[1:]) if len(tokens) > 1 else text
                return text

            text1_no_kunya = _strip_kunya_prefix(text1_norm)
            text2_no_kunya = _strip_kunya_prefix(text2_norm)

            # Recalculate if stripping changed something
            if text1_no_kunya != text1_norm or text2_no_kunya != text2_norm:
                ratio_no_kunya = _char_ratio(text1_no_kunya, text2_no_kunya) / 100.0
                if ratio_no_kunya > ratio1:
                    ratio1 = ratio_no_kunya  # Use the better ratio

        # Fix 24.2: Containment Boost
        # If token_set_ratio indicates high-confidence subset match (>=90%),
        # trust it more heavily to overcome the length penalty from fuzz.ratio.
        # Fix 29.18: Weighted scoring with containment + subset boost.
        # Fix 32.1b: Hard-cap singleton scores at 0.85. A singleton like "אחמד"
        # must NEVER score ≥0.92 against any multi-token contact, regardless of
        # which scoring path fires. The 0.85 cap is below the 0.92 entity_id
        # bypass threshold while still allowing phonebook name resolution (≥0.75).
        # Also catches concatenated singletons ("אחמדכהן") on the ≥0.90 path,
        # and long singletons vs short-extra-token contacts ("אבראהים" vs "אבראהים א").
        def _weighted_score(ratio, token_ratio, t1_norm, t2_norm):
            t1 = set(t1_norm.split())
            t2 = set(t2_norm.split())
            shorter_len = min(len(t1), len(t2))

            # Compute weighted score based on token overlap quality
            if token_ratio >= 0.99:
                if t1.issubset(t2) or t2.issubset(t1):
                    if shorter_len >= 2:
                        score = 0.1 * ratio + 0.9 * token_ratio
                    else:
                        score = 0.2 * ratio + 0.8 * token_ratio
                else:
                    score = 0.2 * ratio + 0.8 * token_ratio
            elif token_ratio >= 0.90:
                score = 0.2 * ratio + 0.8 * token_ratio
            else:
                score = 0.5 * ratio + 0.5 * token_ratio

            # Fix 32.1b: Singleton hard cap — applies to ALL paths above.
            # A singleton must NEVER score ≥0.92 against any contact.
            if shorter_len <= 1:
                return min(0.85, score)
            return score

        score1 = _weighted_score(ratio1, token_ratio1, text1_norm, text2_norm)

        # Score with prefix stripped (for "Al-Mahmoud" vs "Mahmoud")
        if text1_stripped and text2_stripped:
            ratio2 = _char_ratio(text1_stripped, text2_stripped) / 100.0
            token_ratio2 = _token_set_ratio(text1_stripped, text2_stripped) / 100.0
            score2 = _weighted_score(ratio2, token_ratio2, text1_stripped, text2_stripped)
        else:
            score2 = score1

        # Fix 29.11: Per-token al-prefix stripping
        # "אחמד אלברביר" → "אחמד ברביר" (strip אל from each token individually)
        def _strip_al_per_token(text: str) -> str:
            tokens = text.split()
            return ' '.join(t[2:] if t.startswith('אל') and len(t) > 2 else t for t in tokens)

        text1_tok_stripped = _strip_al_per_token(text1_norm)
        text2_tok_stripped = _strip_al_per_token(text2_norm)
        if text1_tok_stripped != text1_norm or text2_tok_stripped != text2_norm:
            ratio3 = _char_ratio(text1_tok_stripped, text2_tok_stripped) / 100.0
            token_ratio3 = _token_set_ratio(text1_tok_stripped, text2_tok_stripped) / 100.0
            score3 = _weighted_score(ratio3, token_ratio3, text1_tok_stripped, text2_tok_stripped)
            return max(score1, score2, score3)
        return max(score1, score2)

# ============================================================================
# UNION-FIND FOR GLOBAL ENTITY ID (Fix 2.2)
# ============================================================================

class UnionFind:
    """Union-Find data structure for merging cross-phone clusters."""

    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            # Deterministic: always use lexicographically smaller as parent
            if px < py:
                self.parent[py] = px
            else:
                self.parent[px] = py

    def get_groups(self) -> Dict[str, List[str]]:
        """Return all groups as {root: [members]}."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for x in self.parent:
            root = self.find(x)
            groups[root].append(x)
        return groups


# ============================================================================

# ============================================================================
# HIERARCHICAL AGGLOMERATIVE CLUSTERING (HAC) - DETERMINISTIC
# ============================================================================

class HacClusterer:
    """Deterministic clustering using Agglomerative Hierarchical Clustering.

    Replaces Chinese Whispers (randomized label propagation) with HAC over a
    precomputed distance matrix.

    Design goals (v7/v8):
    - Determinism: identical results every run (no RNG / seed dependency).
    - Bridge resistance: linkage='average' (default) or 'complete' prevents chaining.
    - Proactive constraints: verified conflicts + BLMZ isolation are enforced in-matrix.

    Notes about this codebase:
    - The similarity graph is *sparse* (edges only above a threshold). For missing edges,
      we treat distance as 1.0 (max). This makes HAC conservative (higher precision).
    - If you need more recall, either (a) lower SIMILARITY_THRESHOLD, or (b) compute a
      full pairwise similarity matrix instead of assuming non-edges are distance=1.0.
    """

    INFINITY: float = 100.0
    FORCE_LINK: float = 1e-3

    def __init__(self, config: Config):
        self.config = config

    def cluster_with_constraints(
        self,
        graph: nx.Graph,
        mentions: List[NameMention]
    ) -> Dict[str, str]:
        """Run HAC on a phone graph, enforcing hard constraints in the distance matrix."""

        # Stable ordering => stable label assignment across runs
        mention_ids = sorted(list(graph.nodes()))
        n = len(mention_ids)

        # Safety guard: HAC builds an O(n^2) distance matrix and (for average/complete linkage)
        # can be slow for very large n. Per-phone n is usually small, but we fail safe.
        max_n = getattr(self.config, "HAC_MAX_N", 800)
        if n > max_n:
            return self._fallback_cluster(graph, mentions, mention_ids)

        if n == 0:
            return {}
        if n == 1:
            return {mention_ids[0]: "hac_0"}

        id_to_idx = {mid: i for i, mid in enumerate(mention_ids)}
        mention_map = {m.mention_id: m for m in mentions}

        # Pre-fetch constraint signals (O(n))
        v_names: List[Optional[str]] = [None] * n
        v_eids: List[Optional[str]] = [None] * n
        blmz_flags: List[bool] = [False] * n

        for mid, i in id_to_idx.items():
            m = mention_map.get(mid)
            if not m:
                continue
            if m.verified_name:
                v_names[i] = m.verified_name
            if m.verified_entity_id:
                v_eids[i] = m.verified_entity_id
            if m.is_blmz:
                blmz_flags[i] = True

        # 1) Build distance matrix from graph edges: distance = 1 - similarity
        dist = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(dist, 0.0)

        for u, v, data in graph.edges(data=True):
            i, j = id_to_idx[u], id_to_idx[v]
            sim = float(data.get('weight', 0.0) or 0.0)
            d = max(0.0, 1.0 - sim)
            dist[i, j] = d
            dist[j, i] = d

        # 2) Inject hard constraints
        INF = self.INFINITY
        EPS = self.FORCE_LINK

        # Enforce per-mention cannot-link constraints (e.g., non-alias segments)
        # This is stricter than relying on "no edge" and remains correct if thresholds change.
        for mid, i in id_to_idx.items():
            m = mention_map.get(mid)
            if not m:
                continue
            for other_id in (getattr(m, 'must_not_link', None) or []):
                j = id_to_idx.get(other_id)
                if j is None:
                    continue
                dist[i, j] = INF
                dist[j, i] = INF


        for i in range(n):
            for j in range(i + 1, n):
                # BLMZ isolation: never merge unknown-speaker mentions with anything
                if blmz_flags[i] or blmz_flags[j]:
                    dist[i, j] = INF
                    dist[j, i] = INF
                    continue

                # Verified entity_id: strongest constraint
                if v_eids[i] and v_eids[j]:
                    if v_eids[i] == v_eids[j]:
                        dist[i, j] = min(dist[i, j], EPS)
                        dist[j, i] = dist[i, j]
                    else:
                        dist[i, j] = INF
                        dist[j, i] = INF
                    continue

                # Verified name: keep your existing "anchor" semantics
                if v_names[i] and v_names[j]:
                    if v_names[i] == v_names[j]:
                        dist[i, j] = min(dist[i, j], EPS)
                        dist[j, i] = dist[i, j]
                    else:
                        dist[i, j] = INF
                        dist[j, i] = INF
                        continue

        # 3) Run clustering
        # Default to average linkage (usually best ER tradeoff). Allow override via config.
        linkage = getattr(self.config, "HAC_LINKAGE", "average")
        # Convert similarity threshold to distance threshold
        threshold = 1.0 - float(self.config.SIMILARITY_THRESHOLD)

        # sklearn API note: older versions use `affinity` instead of `metric`.
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage=linkage,
                distance_threshold=threshold
            )
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                affinity='precomputed',
                linkage=linkage,
                distance_threshold=threshold
            )

        labels_arr = clusterer.fit_predict(dist)

        # 4) Remap sklearn labels -> deterministic, readable IDs
        # sklearn's numeric labels are deterministic given the input order, but we additionally
        # remap based on the smallest mention_id in each cluster to avoid any surprises if
        # internal label numbering changes between sklearn versions.
        by_label: Dict[int, List[str]] = defaultdict(list)
        for mid, lab in zip(mention_ids, labels_arr):
            by_label[int(lab)].append(mid)

        ordered = sorted(
            by_label.items(),
            key=lambda kv: sorted(kv[1])[0]
        )
        label_map = {old: new for new, (old, _) in enumerate(ordered)}

        out: Dict[str, str] = {}
        for mid, lab in zip(mention_ids, labels_arr):
            out[mid] = f"hac_{label_map[int(lab)]}"

        return out

# CLUSTER DSU WITH CONSTRAINT ENFORCEMENT (v8 Architecture)
# ============================================================================


    def _fallback_cluster(
        self,
        graph: nx.Graph,
        mentions: List[NameMention],
        mention_ids: List[str],
    ) -> Dict[str, str]:
        """Deterministic fallback for very large phone graphs.

        HAC (average/complete linkage) requires an O(n^2) distance matrix and can become
        slow for large n. When this happens, we fall back to a deterministic, conservative
        DSU merge on *very high* similarity edges, while still enforcing the same hard
        constraints (BLMZ isolation, verified id/name conflicts, must_not_link).

        This is designed as a fail-safe: it prioritizes precision and stability over recall.
        """
        mention_map = {m.mention_id: m for m in mentions}

        # Precompute constraints
        v_names: Dict[str, str] = {}
        v_eids: Dict[str, str] = {}
        blmz: Set[str] = set()
        cannot: Dict[str, Set[str]] = defaultdict(set)

        for mid in mention_ids:
            m = mention_map.get(mid)
            if not m:
                continue
            if m.verified_name:
                v_names[mid] = m.verified_name
            if m.verified_entity_id:
                v_eids[mid] = m.verified_entity_id
            if m.is_blmz:
                blmz.add(mid)
            for other in (getattr(m, "must_not_link", None) or []):
                cannot[mid].add(other)

        # DSU with component-level constraint tracking to prevent transitive constraint violations
        parent = {mid: mid for mid in mention_ids}
        rank = {mid: 0 for mid in mention_ids}
        # Track blocked mentions and component members per root for transitive constraint checking
        component_blocked: Dict[str, Set[str]] = {mid: set(cannot.get(mid, set())) for mid in mention_ids}
        component_members: Dict[str, Set[str]] = {mid: {mid} for mid in mention_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            # Transitive constraint check: ensure no member of ra's component blocks rb's component
            members_a = component_members.get(ra, {ra})
            members_b = component_members.get(rb, {rb})
            blocked_a = component_blocked.get(ra, set())
            blocked_b = component_blocked.get(rb, set())
            # If any member of A is blocked by B's component, or vice versa, reject merge
            if members_a & blocked_b or members_b & blocked_a:
                return False
            # Perform union
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            # Merge component metadata into new root
            component_members[ra] = members_a | members_b
            component_blocked[ra] = blocked_a | blocked_b
            return True

        # Force-link by verified_entity_id (strongest) and verified_name (secondary)
        by_eid: Dict[str, List[str]] = defaultdict(list)
        by_vname: Dict[str, List[str]] = defaultdict(list)
        for mid, eid in v_eids.items():
            if mid not in blmz:
                by_eid[eid].append(mid)
        for mid, vn in v_names.items():
            if mid not in blmz and mid not in v_eids:
                by_vname[vn].append(mid)

        for eid in sorted(by_eid.keys()):
            ids = sorted(by_eid[eid])
            if len(ids) >= 2:
                first = ids[0]
                for mid in ids[1:]:
                    union(first, mid)

        for vn in sorted(by_vname.keys()):
            ids = sorted(by_vname[vn])
            if len(ids) >= 2:
                first = ids[0]
                for mid in ids[1:]:
                    union(first, mid)

        # Conservative merge threshold for fallback
        base = float(self.config.SIMILARITY_THRESHOLD)
        fallback_sim = getattr(self.config, "HAC_FALLBACK_SIM_THRESHOLD", min(0.92, base + 0.15))

        # Deterministic edge processing
        edges = []
        for u, v, data in graph.edges(data=True):
            sim = float(data.get("weight", 0.0) or 0.0)
            if sim >= fallback_sim:
                uu, vv = (u, v) if u <= v else (v, u)
                edges.append((sim, uu, vv))
        edges.sort(key=lambda t: (-t[0], t[1], t[2]))

        def allowed(u: str, v: str) -> bool:
            # BLMZ never merges
            if u in blmz or v in blmz:
                return False

            # Verified entity id conflicts never merge
            eu, ev = v_eids.get(u), v_eids.get(v)
            if eu and ev and eu != ev:
                return False

            # Verified name conflicts never merge (when both present)
            nu, nv = v_names.get(u), v_names.get(v)
            if nu and nv and nu != nv:
                return False

            # must_not_link blocks unless overridden by verified id/name equality
            if v in cannot.get(u, set()) or u in cannot.get(v, set()):
                if (eu and ev and eu == ev) or (nu and nv and nu == nv):
                    return True
                return False

            return True

        for _, u, v in edges:
            if allowed(u, v):
                union(u, v)

        # Build groups
        groups: Dict[str, List[str]] = defaultdict(list)
        for mid in mention_ids:
            groups[find(mid)].append(mid)

        # Deterministic labeling by smallest mention_id
        ordered = sorted((sorted(members) for members in groups.values()), key=lambda m: m[0])
        out: Dict[str, str] = {}
        for idx, members in enumerate(ordered):
            cid = f"hac_fallback_{idx}"
            for mid in members:
                out[mid] = cid
        return out


class ClusterDSU:
    """Union-Find for clusters with constraint enforcement DURING merge.

    Key improvements over simple UnionFind:
    - Tracks component_entity_ids to enforce entity_id conflicts
    - Tracks component_blocked for cannot-link constraints
    - Tracks merge_reasons for observability
    - Applies cohesion gate to prevent single-linkage bridge collapse
    """

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}  # For union by rank
        self.component_entity_ids: Dict[str, Set[str]] = {}  # root → set of entity_ids
        self.component_members: Dict[str, Set[str]] = {}  # root → set of cluster_ids
        self.component_blocked: Dict[str, Set[str]] = {}  # root → set of blocked cluster_ids
        self.merge_reasons: Dict[str, str] = {}  # cluster_id → reason it was merged
        self.blocked_merges: List[Dict[str, Any]] = []  # For observability

    def make_set(self, cluster_id: str, entity_id: Optional[str] = None, blocked: Set[str] = None) -> None:
        """Initialize a cluster in DSU with its constraints."""
        if cluster_id in self.parent:
            return  # Already exists

        self.parent[cluster_id] = cluster_id
        self.rank[cluster_id] = 0
        self.component_members[cluster_id] = {cluster_id}

        if entity_id:
            self.component_entity_ids[cluster_id] = {entity_id}
        else:
            self.component_entity_ids[cluster_id] = set()

        self.component_blocked[cluster_id] = blocked.copy() if blocked else set()

    def find(self, x: str) -> str:
        """Find with path compression."""
        if x not in self.parent:
            self.make_set(x)

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def can_merge(self, cid1: str, cid2: str) -> Tuple[bool, str]:
        """Check if merging two clusters would violate constraints.

        Returns (can_merge, reason).
        """
        root1 = self.find(cid1)
        root2 = self.find(cid2)

        if root1 == root2:
            return True, 'ALREADY_MERGED'

        # Entity ID conflict check
        ids1 = self.component_entity_ids.get(root1, set())
        ids2 = self.component_entity_ids.get(root2, set())
        merged_ids = ids1 | ids2
        if len(merged_ids) > 1:
            return False, 'ENTITY_ID_CONFLICT'

        # Cannot-link check
        blocked1 = self.component_blocked.get(root1, set())
        blocked2 = self.component_blocked.get(root2, set())
        members1 = self.component_members.get(root1, set())
        members2 = self.component_members.get(root2, set())

        if members1 & blocked2 or members2 & blocked1:
            return False, 'CANNOT_LINK'

        return True, 'OK'

    def union(self, cid1: str, cid2: str, reason: str, score: float = 0.0) -> bool:
        """Merge clusters if constraints allow.

        Returns True if merge succeeded, False if blocked.
        """
        can, why = self.can_merge(cid1, cid2)

        if not can:
            # Record blocked merge for observability
            self.blocked_merges.append({
                'cluster1': cid1,
                'cluster2': cid2,
                'score': score,
                'blocked_by': why,
                'reason_attempted': reason,
            })
            return False

        if why == 'ALREADY_MERGED':
            return True  # No-op

        root1 = self.find(cid1)
        root2 = self.find(cid2)

        # Union by rank
        if self.rank[root1] < self.rank[root2]:
            root1, root2 = root2, root1

        self.parent[root2] = root1
        if self.rank[root1] == self.rank[root2]:
            self.rank[root1] += 1

        # Merge constraint metadata
        self.component_entity_ids[root1] = (
            self.component_entity_ids.get(root1, set()) |
            self.component_entity_ids.get(root2, set())
        )
        self.component_members[root1] = (
            self.component_members.get(root1, set()) |
            self.component_members.get(root2, set())
        )
        self.component_blocked[root1] = (
            self.component_blocked.get(root1, set()) |
            self.component_blocked.get(root2, set())
        )

        # Track merge reason
        self.merge_reasons[cid2] = reason
        if cid1 not in self.merge_reasons:
            self.merge_reasons[cid1] = reason  # First merge reason

        return True

    def get_groups(self) -> Dict[str, List[str]]:
        """Return all groups as {root: [members]}."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for cid in self.parent:
            root = self.find(cid)
            groups[root].append(cid)
        return groups

    def get_global_entity_id(self, cluster_id: str, prefix: str = 'E') -> str:
        """Get deterministic global entity ID for a cluster.

        NOTE: Avoid Python's built-in hash(), which is salted per process
        (PYTHONHASHSEED) and would make IDs change across runs.
        """
        root = self.find(cluster_id)
        import hashlib
        digest = hashlib.sha1(root.encode('utf-8')).hexdigest()[:10]
        num = int(digest, 16) % 100000
        return f"{prefix}{num:05d}"


def cohesion_gate_passes(
    cluster1: 'EntityCluster',
    cluster2: 'EntityCluster',
    scorer: 'SimilarityScorer' = None,
    top_k: int = 3,
    high_threshold: float = 0.92,
    mean_threshold: float = 0.85,
) -> Tuple[bool, float, float]:
    """Check if two clusters have enough cohesion to merge safely.

    Prevents single-linkage bridge collapse by requiring either:
    - One very strong edge (max >= high_threshold), OR
    - Multiple strong edges with anchor (mean(top_k) >= mean_threshold AND has anchor)

    v8 Fix C enhancement: When scorer is used and clusters involve singletons,
    the scorer's built-in singleton cap (0.65-0.80) limits max scores.
    We adjust thresholds accordingly for fair comparison.

    Returns (passes, max_score, mean_top_k_score).
    """
    import statistics

    # v8 Fix C: Detect singleton-dominated clusters
    # The scorer caps singleton-to-multi-token at 0.65 (or 0.80 if unambiguous)
    # So we need lower thresholds when singletons are involved
    def is_singleton_cluster(cluster: 'EntityCluster') -> bool:
        """Check if cluster is dominated by singleton mentions."""
        if not cluster.mentions:
            return False
        singleton_count = sum(1 for m in cluster.mentions if len(m.tokens) == 1)
        return singleton_count > len(cluster.mentions) / 2

    has_singleton = is_singleton_cluster(cluster1) or is_singleton_cluster(cluster2)

    # Adjust thresholds when scorer is used with singleton clusters
    # Scorer caps singletons at 0.65-0.80, so we can't expect 0.92
    if scorer and has_singleton:
        effective_high = 0.60  # Just above the singleton cap
        effective_mean = 0.50  # Lower mean threshold for singletons
    else:
        effective_high = high_threshold
        effective_mean = mean_threshold

    # Fix 27.7: Kunya-Only Bridge Prevention
    # Kunyas like "Abu Ahmed" (Father of Ahmed) are common honorifics that don't
    # uniquely identify a person. Two unrelated people (e.g., "Mahmoud Agbaria"
    # and "Khalid Jabarin") might both be called "Abu Ahmed" because they both
    # have a son named Ahmed. A 100% kunya match should NOT pass cohesion gate.
    # Kunya = Abu (אבו) = "father of" ONLY
    KUNYA_PREFIXES = ('אבו', 'אבו-')

    def _is_kunya_only_mention(m: 'NameMention') -> bool:
        """Check if mention is ONLY a kunya (no family/given name).

        Fix 28.4 (Finding D): Check ALL token positions for kunya patterns, not just first.
        Examples that were previously missed:
        - "מחמד אבו-אחמד" - kunya at position 1
        - "אחמד אל-פג׳לי אבו סאמי" - kunya at positions 2-3
        """
        toks = m.tokens
        if not toks:
            return False

        # Count how many non-kunya tokens exist
        non_kunya_tokens = 0
        i = 0
        while i < len(toks):
            tok = toks[i]
            # Check for standalone kunya prefix
            if tok in KUNYA_PREFIXES:
                # "אבו" followed by name = 2 tokens consumed by kunya
                i += 2  # Skip the kunya and its complement
                continue
            # Check for hyphenated kunya prefix
            if any(tok.startswith(p) for p in KUNYA_PREFIXES if p.endswith('-')):
                # "אבו-אחמד" = 1 token consumed by kunya
                i += 1
                continue
            # This is a non-kunya token
            non_kunya_tokens += 1
            i += 1

        # Mention is kunya-only if it has no non-kunya tokens
        return non_kunya_tokens == 0

    # If scorer not provided, fall back to simple token overlap
    scores = []
    score_details = []  # Track (score, is_kunya_match, is_kunya_dominant) for Fix 27.7 + Fix 29.6

    # Expert Fix 3: Overlap-based kunya guard replacing Fix 29.6
    # Identifies kunya token PAIRS (אבו + complement) so unhyphenated
    # "אבו אחמד" correctly marks "אחמד" as part of kunya expression.
    # Defined outside nested loop to avoid re-creation per iteration.
    def _get_kunya_token_set(tokens):
        result = set()
        i = 0
        while i < len(tokens):
            if tokens[i].startswith('אבו-'):
                result.add(tokens[i])
                i += 1
            elif tokens[i] == 'אבו' and i + 1 < len(tokens):
                result.add(tokens[i])
                result.add(tokens[i + 1])  # complement is also kunya
                i += 2
            else:
                i += 1
        return result

    for m1 in cluster1.mentions:
        for m2 in cluster2.mentions:
            if scorer:
                score = scorer.compute(m1, m2)
            else:
                # Simple fallback: token set overlap
                t1 = set(m1.tokens)
                t2 = set(m2.tokens)
                if t1 and t2:
                    score = len(t1 & t2) / min(len(t1), len(t2))
                else:
                    score = 0.0
            scores.append(score)
            # Fix 27.7: Track if this is a kunya-to-kunya match
            is_kunya_match = _is_kunya_only_mention(m1) and _is_kunya_only_mention(m2)
            kunya_toks_1 = _get_kunya_token_set(m1.tokens)
            kunya_toks_2 = _get_kunya_token_set(m2.tokens)
            shared_tokens = set(m1.tokens) & set(m2.tokens)
            is_kunya_dominant = (len(shared_tokens) > 0 and
                                 shared_tokens <= (kunya_toks_1 | kunya_toks_2))
            score_details.append((score, is_kunya_match, is_kunya_dominant))

    if not scores:
        return False, 0.0, 0.0

    scores.sort(reverse=True)
    max_score = scores[0]
    top_k_scores = scores[:min(top_k, len(scores))]
    mean_score = statistics.mean(top_k_scores) if top_k_scores else 0.0

    # Fix 27.7: Check if ALL high-scoring pairs are kunya-only matches
    # A high score (>= high_threshold) from kunya-to-kunya is "false entropy"
    kunya_bridge_threshold = 0.85  # Pairs above this are checked for kunya-only
    high_scoring_pairs = [(s, is_k, is_d) for s, is_k, is_d in score_details if s >= kunya_bridge_threshold]

    if high_scoring_pairs:
        # Check if ALL high-scoring pairs are kunya-only
        all_kunya_only = all(is_k for _, is_k, _ in high_scoring_pairs)
        # Fix 29.6: Also check kunya-dominant (shared tokens are all kunya, even if mentions have non-kunya tokens)
        all_kunya_bridge = all(is_k or is_d for _, is_k, is_d in high_scoring_pairs)
        if all_kunya_only or all_kunya_bridge:
            # Additional check: are there ANY non-kunya pairs with moderate scores?
            non_kunya_threshold = 0.30
            has_non_kunya_evidence = any(
                s >= non_kunya_threshold for s, is_k, is_d in score_details if not is_k and not is_d
            )
            if not has_non_kunya_evidence:
                # BLOCK: Only kunya matches with no supporting evidence
                return False, max_score, mean_score

    # FIX: Prevent "bridge collapse" by checking score density for large clusters.
    # The top-k mean alone can be fooled by a few noise matches (e.g., 3 "David"s
    # scoring 1.0 among 10,000 pairs). For large comparisons, require that a
    # minimum percentage of pairs have non-trivial similarity.
    total_pairs = len(scores)
    if total_pairs > top_k * 3:  # Only apply density check for large comparisons
        # Count pairs with non-trivial similarity (above noise threshold)
        noise_threshold = 0.15
        nonzero_count = sum(1 for s in scores if s > noise_threshold)
        density_ratio = nonzero_count / total_pairs
        # Require at least 5% of pairs to have some similarity for large clusters
        # This prevents 3 noise matches from bridging 10,000-pair comparisons
        min_density = 0.05
        if density_ratio < min_density:
            # Not enough overall similarity - likely a noise bridge
            return False, max_score, mean_score

    # Check if either cluster is a *strong* anchor.
    # v9: PHONEBOOK is only an anchor when the matched contact is high-quality.
    def _is_strong_anchor(c: 'EntityCluster') -> bool:
        if c.verified_entity_id:
            return True
        if c.resolution_type == 'CALL_VERIFIED' or c.has_verified():
            return True
        if c.resolution_type == 'PHONEBOOK':
            return (getattr(c, 'phonebook_quality', '') or '') in ('HIGH',)
        return False

    has_anchor = _is_strong_anchor(cluster1) or _is_strong_anchor(cluster2)

    # Gate logic with adjusted thresholds
    if max_score >= effective_high:
        return True, max_score, mean_score

    if mean_score >= effective_mean and has_anchor:
        return True, max_score, mean_score

    return False, max_score, mean_score


# ============================================================================
# RESOLUTION
# ============================================================================

class EntityResolver:
    """Resolve clusters to canonical entities using priority cascade."""

    def __init__(self, config: Config, cube2_matcher: Cube2Matcher, normalizer: Normalizer = None,
                 ambiguity_gate: Optional[AmbiguityGate] = None):
        self.config = config
        self.cube2_matcher = cube2_matcher
        self.normalizer = normalizer or Normalizer(config)
        self.noise_tokens = config.get_noise_tokens_normalized()
        self.blocked_merges: List[Dict[str, Any]] = []  # v8: Track blocked merges for observability
        # Fix 13.2: Store ambiguity_gate for use in resolve_cluster
        self.ambiguity_gate = ambiguity_gate

    @staticmethod
    def _extract_family_token(tokens: List[str]) -> Optional[str]:
        """Extract family token, handling kunya-at-end cases (Fix 27.3).

        Fix 28.5b: Minimum length checks match _detect_nickname_hijack.
        For 2-token mentions with kunya-at-end like ["מחמד", "אבו-אחמד"],
        no family is available, so returns None.
        """
        if not tokens or len(tokens) < 2:
            return None
        if tokens[-1].startswith('אבו-'):
            return normalize_arabic_phonetic(tokens[-2]) if len(tokens) >= 3 else None
        if tokens[-2] == 'אבו':
            return normalize_arabic_phonetic(tokens[-3]) if len(tokens) >= 4 else None
        for tok in reversed(tokens):
            if not tok.startswith('אבו') and tok not in ('אבו',):
                return normalize_arabic_phonetic(tok)
        return None

    # ------------------------------------------------------------------
    # cube2 confidence + quality helpers (v9)
    # ------------------------------------------------------------------
    def _quality_rank(self, tier: str) -> int:
        tier = (tier or '').upper().strip()
        return Cube2Matcher.QUALITY_RANK.get(tier, 0)

    def _cube2_is_confident(self, match: Optional[Cube2Match], *, min_margin: Optional[float] = None) -> bool:
        """Return True if a cube2 match is strong enough to be used as evidence.

        Fix 29.9: When the match carries an entity_id, use a relaxed margin
        threshold.  Entity_id already confirms identity so the margin guard
        (designed to distinguish two ambiguous contacts) is largely redundant.
        At scale (400+ records) many similar Arabic names compress margin below
        0.15 even for correct matches, causing false rejections.
        """
        if not match:
            return False
        if min_margin is not None:
            margin_th = float(min_margin)
        elif getattr(match, 'entity_id', None):
            margin_th = self.config.CUBE2_MARGIN_THRESHOLD_WITH_ENTITY_ID
        else:
            margin_th = self.config.CUBE2_MARGIN_THRESHOLD
        return match.margin >= margin_th

    def _cube2_phonebook_eligible(self, match: Cube2Match, phone: str = '') -> bool:
        """Return True if this cube2 match can upgrade a cluster to PHONEBOOK."""
        if not match:
            return False
        # Entity_id alone is NOT enough - must also pass quality gate.
        # LOW quality contacts like "Mom", "Taxi" should NOT become PHONEBOOK anchors
        # even if they have an entity_id, as this causes "black hole" entity merges.

        # Fix 14.1: IDF-aware quality gating to prevent "Black Hole" contacts
        # Contacts with very common names (low mean_idf) should be demoted even if
        # they pass generic token filtering. E.g., "David Work" has HIGH tier but
        # "David" is extremely common, creating a black hole for all Davids.
        effective_tier = match.quality_tier
        if phone and match.contact_key:
            contact = self.cube2_matcher.get_contact_by_key(phone, match.contact_key)
            if contact:
                # Fix 29.22: Default to 5.0 to match finalize_contact_quality().
                # Previously 1.0, which incorrectly triggers demotion when IDF
                # data is unavailable (contact tokens not in cube1 vocabulary).
                mean_idf = contact.get('mean_idf', 5.0)
                # Fix 21.1 + Fix 29.10: IDF demotion with configurable threshold
                if mean_idf < self.config.IDF_DEMOTION_THRESHOLD and effective_tier == 'HIGH':
                    effective_tier = 'MED'

        min_tier = getattr(self.config, 'CUBE2_PHONEBOOK_MIN_QUALITY_TIER_FOR_RESOLUTION', 'HIGH')
        quality_ok = self._quality_rank(effective_tier) >= self._quality_rank(min_tier)
        # Allow entity_id to lower the bar to MED (but not LOW)
        if match.entity_id and self._quality_rank(effective_tier) >= self._quality_rank('MED'):
            return True
        return quality_ok

    @staticmethod
    def _cube2_without_entity_id(match: Optional[Cube2Match]) -> Optional[Cube2Match]:
        """Return a copy of cube2 match without hard entity_id attachment."""
        if not match:
            return None
        if not getattr(match, 'entity_id', None):
            return match
        return Cube2Match(
            name=match.name,
            nickname=match.nickname,
            score=float(match.score or 0.0),
            second_score=float(match.second_score or 0.0),
            margin=float(match.margin or 0.0),
            entity_id=None,
            contact_key=match.contact_key,
            name_normalized=match.name_normalized,
            tokens=list(match.tokens or []),
            quality_tier=match.quality_tier,
            quality_score=float(match.quality_score or 0.0),
            source_phone=getattr(match, 'source_phone', None),
            status=match.status,
            id_number=match.id_number,
        )

    @staticmethod
    def _is_kunya_token(tok: str) -> bool:
        return bool(tok) and (tok == 'אבו' or tok.startswith('אבו-'))

    def _extract_kunya_tokens_from_text(self, text: str) -> Set[str]:
        out: Set[str] = set()
        if not text:
            return out
        normalized = self.normalizer.normalize(str(text))
        toks = self.normalizer.tokenize(normalized)
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == 'אבו' and i + 1 < len(toks):
                out.add(normalize_kunya_alias_token(f'אבו-{toks[i + 1]}'))
                i += 2
                continue
            if tok.startswith('אבו-'):
                out.add(normalize_kunya_alias_token(tok))
            i += 1
        out.discard('')
        return out

    def _collect_cluster_kunya_tokens(self, mentions: List[NameMention]) -> Set[str]:
        out: Set[str] = set()
        for m in mentions:
            toks = m.tokens or []
            i = 0
            while i < len(toks):
                tok = toks[i]
                if tok == 'אבו' and i + 1 < len(toks):
                    out.add(normalize_kunya_alias_token(f'אבו-{toks[i + 1]}'))
                    i += 2
                    continue
                if tok.startswith('אבו-'):
                    out.add(normalize_kunya_alias_token(tok))
                i += 1
        out.discard('')
        return out

    def _contact_kunya_tokens(self, contact: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        if not contact:
            return out

        toks = list(contact.get('tokens') or [])
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == 'אבו' and i + 1 < len(toks):
                out.add(normalize_kunya_alias_token(f'אבו-{toks[i + 1]}'))
                i += 2
                continue
            if tok.startswith('אבו-'):
                out.add(normalize_kunya_alias_token(tok))
            i += 1

        for nick in (contact.get('nicknames_normalized') or []):
            out.update(self._extract_kunya_tokens_from_text(nick))

        out.discard('')
        return out

    @staticmethod
    def _is_negation_ambiguity_token(tok: str) -> bool:
        return bool(tok) and tok in {'לא', 'בלי'}

    @staticmethod
    def _is_contrast_ambiguity_token(tok: str) -> bool:
        return bool(tok) and tok in {'אלא', 'או'}

    def _mention_has_negation_ambiguity(self, mention: NameMention) -> bool:
        toks = [str(tok or '').strip() for tok in (mention.tokens or []) if str(tok or '').strip()]
        if len(toks) < 4:
            return False
        has_neg = any(self._is_negation_ambiguity_token(tok) for tok in toks)
        has_contrast = any(self._is_contrast_ambiguity_token(tok) for tok in toks)
        return has_neg and has_contrast

    def _singleton_context_hard_block_reason(
        self,
        cluster: EntityCluster,
        cube2_match: Optional[Cube2Match],
    ) -> str:
        """Block hard assignment on singleton negation/list/repetition ambiguity."""
        if not getattr(self.config, 'SINGLETON_CONTEXT_HARD_GUARD', True):
            return ''
        if len(cluster.mentions) != 1:
            return ''
        m0 = cluster.mentions[0]
        if str(getattr(m0, 'verified_entity_id', '') or '').strip():
            return ''

        if self._mention_has_negation_ambiguity(m0):
            return 'singleton_negation_ambiguous_blocked'

        non_kunya_tokens = [
            normalize_arabic_phonetic(tok)
            for tok in (m0.tokens or [])
            if tok and (not self._is_kunya_token(tok))
        ]
        non_kunya_tokens = [tok for tok in non_kunya_tokens if tok]
        unique_non_kunya = set(non_kunya_tokens)

        local_margin = float(getattr(cube2_match, 'margin', 0.0) or 0.0)
        local_top = float(getattr(cube2_match, 'score', 0.0) or 0.0)

        min_rep = int(getattr(self.config, 'SINGLETON_REPETITION_MIN_COUNT', 3))
        if len(unique_non_kunya) == 1 and len(non_kunya_tokens) >= min_rep:
            return 'singleton_repetition_ambiguous_blocked'

        min_unique = int(getattr(self.config, 'SINGLETON_LIST_MIN_UNIQUE_TOKENS', 3))
        low_margin = float(getattr(self.config, 'SINGLETON_LOW_MARGIN_THRESHOLD', 0.08))
        low_top = float(getattr(self.config, 'SINGLETON_LOW_TOP_SCORE_THRESHOLD', 0.90))
        if len(unique_non_kunya) >= min_unique and local_margin < low_margin and local_top < low_top:
            return 'singleton_low_margin_ambiguous_blocked'

        return ''

    def _kunya_ambiguity_hard_block_reason(
        self,
        phone: str,
        cluster: EntityCluster,
        cube2_match: Optional[Cube2Match],
    ) -> str:
        """Block hard assignment when kunya-only evidence is globally ambiguous."""
        if not getattr(self.config, 'KUNYA_AMBIGUITY_HARD_GUARD', True):
            return ''
        if not cube2_match or not cube2_match.nickname:
            return ''
        if not self._is_kunya_only_cluster(cluster.mentions):
            return ''

        matched_kunyas = self._extract_kunya_tokens_from_text(cube2_match.nickname)
        if not matched_kunyas:
            return ''

        # Candidate corroboration: repeated kunya mentions can be enough if the
        # top-vs-second margin remains strong.
        multi_mention = len(cluster.mentions) >= 2
        min_margin = max(
            float(self.config.CUBE2_MARGIN_THRESHOLD),
            float(getattr(self.config, 'KUNYA_GUARD_MULTI_MENTION_MIN_MARGIN', 0.20))
        )
        min_score = float(getattr(self.config, 'KUNYA_GUARD_MULTI_MENTION_MIN_SCORE', 0.90))
        strong_multi_consensus = (
            multi_mention and
            float(getattr(cube2_match, 'margin', 0.0) or 0.0) >= min_margin and
            float(getattr(cube2_match, 'score', 0.0) or 0.0) >= min_score
        )

        # When margin is extremely high, the scoring itself has fully
        # disambiguated the kunya — the second-best contact scored near 0.
        # Safe to allow even single mentions through.
        single_high_margin_threshold = float(
            getattr(self.config, 'KUNYA_GUARD_SINGLE_HIGH_MARGIN', 0.80)
        )
        single_high_score_threshold = float(
            getattr(self.config, 'KUNYA_GUARD_SINGLE_HIGH_SCORE', 0.95)
        )
        match_margin = float(getattr(cube2_match, 'margin', 0.0) or 0.0)
        match_score = float(getattr(cube2_match, 'score', 0.0) or 0.0)
        scoring_disambiguated = (
            match_margin >= single_high_margin_threshold and
            match_score >= single_high_score_threshold
        )

        # Detect ambiguity globally (across all phones), not just same phone.
        global_contact_keys: Set[str] = set()
        global_entity_ids: Set[str] = set()
        phone_contact_keys: Set[str] = set()
        phone_entity_ids: Set[str] = set()
        phone_norm = normalize_phone(phone)

        for p, contacts in self.cube2_matcher.contacts_by_phone.items():
            for contact in contacts:
                contact_kunyas = self._contact_kunya_tokens(contact)
                if not contact_kunyas or not (contact_kunyas & matched_kunyas):
                    continue

                ckey = str(contact.get('contact_key') or '').strip()
                if ckey:
                    global_contact_keys.add(ckey)
                    if p == phone_norm:
                        phone_contact_keys.add(ckey)

                eid = _coerce_entity_id(contact.get('entity_id'))
                if eid:
                    global_entity_ids.add(eid)
                    if p == phone_norm:
                        phone_entity_ids.add(eid)

        global_ambiguous = (len(global_contact_keys) >= 2) or (len(global_entity_ids) >= 2)
        phone_ambiguous = (len(phone_contact_keys) >= 2) or (len(phone_entity_ids) >= 2)
        if not (global_ambiguous or phone_ambiguous):
            return ''
        if strong_multi_consensus:
            return ''
        if scoring_disambiguated:
            return ''
        return 'kunya_ambiguous_hard_blocked'

    def _cluster_kunya_tokens(self, cluster: EntityCluster) -> Set[str]:
        """Collect kunya tokens from mentions, nicknames and linked contact."""
        out: Set[str] = set()
        if not cluster:
            return out

        out.update(self._collect_cluster_kunya_tokens(cluster.mentions or []))

        for nick in (cluster.nicknames or set()):
            out.update(self._extract_kunya_tokens_from_text(str(nick)))

        contact_key = getattr(cluster, 'cube2_contact_key', None)
        if contact_key:
            contact = self.cube2_matcher.get_contact_by_key(cluster.phone, contact_key)
            if contact:
                out.update(self._contact_kunya_tokens(contact))

        out.discard('')
        return out

    def _cluster_is_strong_kunya_anchor(self, cluster: EntityCluster) -> bool:
        """Return True when cluster can safely serve as kunya alias anchor."""
        if not cluster or not getattr(cluster, 'verified_entity_id', None):
            return False

        if cluster.resolution_type in ('CALL_VERIFIED', 'VERIFIED_CONFLICT'):
            return True

        # Mention-level verified IDs are strong supervision even if resolution type differs.
        if any(bool(getattr(m, 'verified_entity_id', None)) for m in (cluster.mentions or [])):
            return True

        if cluster.resolution_type == 'PHONEBOOK':
            min_score = float(getattr(self.config, 'KUNYA_ALIAS_PROPAGATION_MIN_SCORE', 0.85))
            min_margin = float(getattr(self.config, 'KUNYA_ALIAS_PROPAGATION_MIN_MARGIN', 0.12))
            return (
                float(getattr(cluster, 'best_score', 0.0) or 0.0) >= min_score and
                float(getattr(cluster, 'score_margin', 0.0) or 0.0) >= min_margin
            )

        return False

    def propagate_kunya_alias_on_phone(self, clusters: List[EntityCluster]) -> List[EntityCluster]:
        """Propagate EID to unresolved kunya-only clusters using same-phone anchors.

        Safety constraints:
        - unresolved cluster must be kunya-only
        - exactly one strong anchor entity on the same phone shares the kunya token(s)
        - skip when same-phone anchor entities are ambiguous for that kunya
        """
        if not getattr(self.config, 'KUNYA_ALIAS_PROPAGATION_ENABLED', True):
            return clusters
        if not clusters:
            return clusters

        clusters_by_phone: Dict[str, List[EntityCluster]] = defaultdict(list)
        for c in clusters:
            clusters_by_phone[c.phone].append(c)

        for _, phone_clusters in clusters_by_phone.items():
            kunya_to_anchor_eids: Dict[str, Set[str]] = defaultdict(set)
            eid_to_metadata: Dict[str, Tuple[Optional[str], Optional[str]]] = {}  # eid -> (status, id_number)

            for c in phone_clusters:
                if not self._cluster_is_strong_kunya_anchor(c):
                    continue
                eid = _coerce_entity_id(getattr(c, 'verified_entity_id', None))
                if not eid:
                    continue
                if eid not in eid_to_metadata:
                    eid_to_metadata[eid] = (c.verified_status, c.verified_id_number)
                ck = self._cluster_kunya_tokens(c)
                for kunya_tok in ck:
                    if kunya_tok:
                        kunya_to_anchor_eids[kunya_tok].add(eid)

            if not kunya_to_anchor_eids:
                continue

            for c in phone_clusters:
                if getattr(c, 'verified_entity_id', None):
                    continue
                if not self._is_kunya_only_cluster(c.mentions or []):
                    continue

                cluster_kunyas = self._collect_cluster_kunya_tokens(c.mentions or [])
                if not cluster_kunyas:
                    continue

                candidate_eids: Set[str] = set()
                has_local_ambiguity = False
                for kunya_tok in cluster_kunyas:
                    eids_for_kunya = kunya_to_anchor_eids.get(kunya_tok, set())
                    if len(eids_for_kunya) >= 2:
                        has_local_ambiguity = True
                    candidate_eids.update(eids_for_kunya)

                if has_local_ambiguity:
                    continue
                if len(candidate_eids) != 1:
                    continue

                chosen_eid = next(iter(candidate_eids))
                c.verified_entity_id = chosen_eid
                anchor_meta = eid_to_metadata.get(chosen_eid, (None, None))
                if not c.verified_status:
                    c.verified_status = anchor_meta[0]
                if not c.verified_id_number:
                    c.verified_id_number = anchor_meta[1]
                if 'kunya_alias_propagated' not in c.flags:
                    c.flags.append('kunya_alias_propagated')
                if c.match_evidence:
                    if 'kunya_alias_propagated' not in c.match_evidence:
                        c.match_evidence = f"{c.match_evidence};kunya_alias_propagated"
                else:
                    c.match_evidence = 'kunya_alias_propagated'

        return clusters

    def resolve_cluster(
        self,
        cluster_id: str,
        phone: str,
        mentions: List[NameMention]
    ) -> EntityCluster:
        """Resolve a cluster to a canonical entity."""
        cluster = EntityCluster(
            cluster_id=cluster_id,
            phone=phone,
            mentions=mentions,
            mention_ids={m.mention_id for m in mentions}
        )

        if self._is_blmz_cluster(mentions):
            return self._resolve_as_blmz(cluster)

        # v8 FIX: Always check cube2 for entity_id, even for CALL_VERIFIED clusters
        # This ensures we capture entity_id from phonebook when cube1 doesn't have it
        cube2_match = self.cube2_matcher.match(phone, mentions)

        # Fix 25.1 + Fix 29.2 + Expert Fix 1: Aggressive global rescue
        # Always check global when local is absent, fails gates, OR is "good enough"
        # but inferior (no entity_id, score < 0.98). A strictly better global match
        # (has entity_id when local doesn't, higher score, or better tier) replaces local.
        should_check_global = False
        local_passes_gates = False
        local_is_kunya_only = False
        if not cube2_match:
            should_check_global = True
        else:
            local_confident = self._cube2_is_confident(cube2_match)
            local_eligible = self._cube2_phonebook_eligible(cube2_match, phone)
            local_passes_gates = local_confident and local_eligible
            if not local_passes_gates:
                # Fix 29.2: Local fails gates - try global rescue
                should_check_global = True
            elif cube2_match.score < 0.98 or not getattr(cube2_match, 'entity_id', None):
                # Expert Fix 1: "Good enough" trap - local passes gates but may be inferior
                should_check_global = True
            elif cube2_match.nickname and self._is_kunya_only_cluster(mentions):
                # Expert Fix 1b: Kunya-evidence trap - local matched via nickname and
                # ALL cluster mentions are kunya-only (e.g., "אבו אחמד"). Even with
                # high score + entity_id, kunya-only evidence is semantically weak
                # (thousands of people share the same kunya). Global check is critical
                # to find a match with real family-name evidence.
                should_check_global = True
                local_is_kunya_only = True

        if should_check_global:
            global_match = self.cube2_matcher.match_global(mentions)
            if global_match:
                global_phone = getattr(global_match, 'source_phone', None) or phone
                global_confident = self._cube2_is_confident(global_match)
                global_eligible = self._cube2_phonebook_eligible(global_match, global_phone)
                if global_confident and global_eligible:
                    if not cube2_match:
                        # No local match - use global
                        cube2_match = global_match
                        cluster.flags.append('global_phonebook_match')
                    elif not local_passes_gates:
                        # Local fails gates - global rescue
                        cube2_match = global_match
                        cluster.flags.append('global_phonebook_rescue')
                    elif local_is_kunya_only and not global_match.nickname:
                        # Expert Fix 1b: Local is kunya-only match, global matched
                        # by main name (not nickname). Main-name evidence is stronger
                        # than nickname/kunya evidence — replace regardless of score.
                        cube2_match = global_match
                        cluster.flags.append('global_phonebook_kunya_upgrade')
                    else:
                        # Local passes gates - replace only if global is strictly better
                        has_better_id = (
                            getattr(global_match, 'entity_id', None) and
                            not getattr(cube2_match, 'entity_id', None)
                        )
                        is_better_score = global_match.score > cube2_match.score + 0.05
                        local_tier = getattr(cube2_match, 'quality_tier', None)
                        global_tier = getattr(global_match, 'quality_tier', None)
                        is_better_tier = (global_tier == 'HIGH' and local_tier != 'HIGH')
                        if has_better_id or is_better_score or is_better_tier:
                            cube2_match = global_match
                            cluster.flags.append('global_phonebook_upgrade')

        verified_result = self._check_verified(mentions)
        if verified_result:
            return self._resolve_as_verified(cluster, verified_result, cube2_match)

        # Fix 29.4: Use source_phone for cross-phone eligibility checks.
        # When cube2_match came from global rescue, the contact lives on a different
        # phone. Using cluster.phone for get_contact_by_key() would fail to find the
        # contact → IDF demotion is SKIPPED → common names could be incorrectly
        # treated as HIGH quality ("black hole" merges).
        eligibility_phone = getattr(cube2_match, 'source_phone', None) or phone if cube2_match else phone

        # Fix 13.2: Apply ambiguity gate check before phonebook resolution
        # This prevents "Yossi" from resolving to "Yossi Cohen" when "Yossi Levi" also exists
        # Uses eligibility_phone so cross-phone matches check ambiguity on the contact's phone
        phonebook_blocked_by_ambiguity = False
        if cube2_match and self.ambiguity_gate:
            phonebook_blocked_by_ambiguity = self._cube2_bridge_blocked_by_ambiguity(
                eligibility_phone, cluster, cube2_match, self.ambiguity_gate
            )

        hard_block_reasons: List[str] = []
        if cube2_match:
            kunya_block_reason = self._kunya_ambiguity_hard_block_reason(
                eligibility_phone, cluster, cube2_match
            )
            if kunya_block_reason:
                hard_block_reasons.append(kunya_block_reason)

            singleton_block_reason = self._singleton_context_hard_block_reason(cluster, cube2_match)
            if singleton_block_reason:
                hard_block_reasons.append(singleton_block_reason)

        for reason in hard_block_reasons:
            if reason not in cluster.flags:
                cluster.flags.append(reason)

        # Fix 32.3: Singleton entity_id guard — when ALL mentions are
        # single-token common names (no family name, no kunya), do NOT
        # propagate entity_id from phonebook. A singleton like "אחמד"
        # could be any of several people sharing that given name.
        # Preserves PHONEBOOK resolution (correct canonical name) but
        # strips entity_id to prevent Phase 0 cascade across phones.
        # Cube1 entity_id (mention-level) is unaffected — extracted inside each path.
        # Kunya mentions (אבו-X) are NOT stripped — they are specific identifiers.
        if cube2_match and cube2_match.entity_id:
            all_singleton = True
            for m in cluster.mentions:
                if not m.tokens:
                    continue
                if len(m.tokens) >= 2:
                    all_singleton = False
                    break
                tok = m.tokens[0]
                if tok.startswith('אבו-') or tok == 'אבו':
                    all_singleton = False  # Kunya = specific, not common name
                    break
            if all_singleton:
                cube2_match = self._cube2_without_entity_id(cube2_match)
                if 'singleton_entity_id_stripped' not in cluster.flags:
                    cluster.flags.append('singleton_entity_id_stripped')

        if (cube2_match and self._cube2_is_confident(cube2_match)
            and self._cube2_phonebook_eligible(cube2_match, eligibility_phone)
            and not phonebook_blocked_by_ambiguity
            and not hard_block_reasons):
            return self._resolve_as_phonebook(cluster, cube2_match)

        cube2_for_inferred = cube2_match
        if cube2_match and (phonebook_blocked_by_ambiguity or hard_block_reasons):
            cube2_for_inferred = self._cube2_without_entity_id(cube2_match)
        return self._resolve_as_inferred(cluster, cube2_for_inferred)

    def _is_blmz_cluster(self, mentions: List[NameMention]) -> bool:
        return all(m.is_blmz for m in mentions)

    @staticmethod
    def _is_kunya_only_cluster(mentions: List[NameMention]) -> bool:
        """Return True if ALL mentions in the cluster are kunya-only (no family/given names).

        Expert Fix 1b: Used to detect when a high-scoring per-phone match is based
        entirely on kunya evidence, which is semantically weak (thousands of people
        share common kunyas like 'Abu Ahmed'). Forces global check for better evidence.
        """
        if not mentions:
            return False
        for m in mentions:
            toks = m.tokens
            if not toks:
                return False  # Empty tokens = unknown, treat conservatively as non-kunya
            # Check if this mention has any non-kunya tokens
            i = 0
            has_non_kunya = False
            while i < len(toks):
                tok = toks[i]
                if tok == 'אבו':
                    i += 2  # standalone kunya + complement
                    continue
                if tok.startswith('אבו-'):
                    i += 1  # hyphenated kunya (e.g., "אבו-אחמד")
                    continue
                has_non_kunya = True
                break
            if has_non_kunya:
                return False
        return True

    def _check_verified(self, mentions: List[NameMention]) -> Optional[Dict]:
        verified_names = []
        nicknames = set()

        for m in mentions:
            if m.verified_name:
                verified_names.append(m.verified_name)
            if m.verified_nicknames:
                for nick in _split_nicknames(m.verified_nicknames):
                    nicknames.add(nick)

        if not verified_names:
            return None

        name_counts = Counter(verified_names)
        canonical, count = name_counts.most_common(1)[0]

        return {
            'canonical': canonical,
            'nicknames': nicknames,
            'unique_names': set(verified_names),
            'total_verified': len(verified_names),
        }

    def _resolve_as_verified(
        self,
        cluster: EntityCluster,
        verified_result: Dict,
        cube2_match: Optional[Cube2Match] = None
    ) -> EntityCluster:
        cluster.canonical_name = verified_result['canonical']
        cluster.display_name = verified_result['canonical']  # Already raw from cube1
        cluster.nicknames = verified_result['nicknames']
        cluster.best_score = 1.0
        cluster.second_best_score = 0.0
        cluster.score_margin = 1.0

        # Extract verified_entity_id from mentions (cube1 - highest priority)
        for m in cluster.mentions:
            if m.verified_entity_id and (not getattr(m, 'is_blmz', False)):
                cluster.verified_entity_id = m.verified_entity_id
                cluster.verified_status = m.verified_status
                cluster.verified_id_number = m.verified_id_number
                break

        # v8 FIX: Fallback to cube2 entity_id if cube1 doesn't have one
        # This ensures we capture entity_id from phonebook when call data lacks it
        # IMPORTANT: Only use cube2 entity_id if the match is confident (names match well)
        # Fix 16.3 + 16.4: Apply nickname safety with promoted-canonical exception
        # A nickname match (e.g., "Taxi" with nickname "Yossi") shouldn't inject entity_id.
        # EXCEPTION: If the nickname is the high-quality one that upgrades contact quality
        # (best_nickname_for_quality), it's trusted and entity_id can propagate.
        if (not cluster.verified_entity_id
            and cube2_match
            and self._cube2_is_confident(cube2_match)
            and cube2_match.entity_id):
            # Check nickname safety with promoted-canonical exception
            nickname_safe = not cube2_match.nickname
            if cube2_match.nickname and cube2_match.contact_key:
                # Fix 44: Use source_phone for cross-phone contact lookups
                lookup_phone = getattr(cube2_match, 'source_phone', None) or cluster.phone
                contact = self.cube2_matcher.get_contact_by_key(lookup_phone, cube2_match.contact_key)
                if contact and contact.get('best_nickname_for_quality') == cube2_match.nickname:
                    nickname_safe = True  # Promoted canonical nickname is trusted

                # Fix 29.6a: Family Corroboration Exception for CALL_VERIFIED path
                # Same logic as phonebook/inferred paths (Fix 28.5): when matched via nickname,
                # allow entity_id if the cluster's family name matches the contact's family name.
                elif contact:
                    cluster_family = None
                    for m in cluster.mentions:
                        if m.tokens and len(m.tokens) >= 2:
                            cluster_family = self._extract_family_token(m.tokens)
                            if cluster_family:
                                break

                    contact_tokens = contact.get('tokens') or []
                    contact_family = self._extract_family_token(contact_tokens)

                    if cluster_family and contact_family:
                        family_match_score = _char_ratio(cluster_family, contact_family)
                        if family_match_score >= 85:
                            nickname_safe = True
            # Fix 32.3b: Singleton entity_id guard for CALL_VERIFIED path.
            # This path returns BEFORE the main Fix 32.3 guard at line ~5332,
            # so singletons must be checked here too.
            all_singleton_mentions = True
            for m in cluster.mentions:
                if not m.tokens:
                    continue
                if len(m.tokens) >= 2:
                    all_singleton_mentions = False
                    break
                tok = m.tokens[0]
                if tok.startswith('אבו-') or tok == 'אבו':
                    all_singleton_mentions = False
                    break
            if all_singleton_mentions:
                nickname_safe = False  # Block entity_id for singletons

            # Fix 29.8: High-score corroboration — score ≥0.92 is strong identity evidence
            if nickname_safe or (cube2_match.score >= 0.92):
                cluster.verified_entity_id = cube2_match.entity_id
                if not cluster.verified_status:
                    cluster.verified_status = cube2_match.status
                if not cluster.verified_id_number:
                    cluster.verified_id_number = cube2_match.id_number

        unique_names = verified_result['unique_names']
        if len(unique_names) > 1:
            cluster.resolution_type = 'VERIFIED_CONFLICT'
            cluster.confidence = 'MED'
            cluster.flags.append('verified_conflict')
            cluster.match_evidence = f"verified_names:{json.dumps(list(unique_names), ensure_ascii=False)}"
        else:
            cluster.resolution_type = 'CALL_VERIFIED'
            cluster.confidence = 'HIGH'
            cluster.match_evidence = f"verified:{cluster.canonical_name}"

        return cluster

    def _resolve_as_phonebook(self, cluster: EntityCluster, cube2_match: Cube2Match) -> EntityCluster:
        # Fix 12.2 + Fix 15.1: Dynamic Canonical Swapping (Always Check)
        # ALWAYS check if a high-quality nickname should replace a low-quality primary name.
        # Previously, we only checked when match was via nickname (cube2_match.nickname was set).
        # Fix 15.1: Now check unconditionally, so "Mom" with nickname "Rivka Cohen" uses "Rivka Cohen".
        canonical_to_use = cube2_match.name
        # Fix 44: Use source_phone for cross-phone contact lookups
        # When match_global() found contact on Phone B but cluster is on Phone A,
        # we must use the contact's source phone to retrieve full contact details.
        lookup_phone = getattr(cube2_match, 'source_phone', None) or cluster.phone
        contact = self.cube2_matcher.get_contact_by_key(lookup_phone, cube2_match.contact_key)
        if contact and contact.get('best_nickname_for_quality'):
            promoted_nick = contact['best_nickname_for_quality']
            # Fix 29.19: Prevent kunya-only nickname promotion to canonical.
            # Kunyas like "אבו-אחמד" are honorifics, not real names. They can get
            # HIGH quality tier due to kunya token counting (Fix 24.3), but should
            # NOT become the canonical name as they don't identify individuals.
            # Fix 29.19b: Handle unhyphenated kunyas like "אבו אחמד" (2 tokens).
            # Must use same logic as _is_kunya_only_mention() in cohesion gate.
            nick_tokens = promoted_nick.split()
            non_kunya_count = 0
            i = 0
            while i < len(nick_tokens):
                tok = nick_tokens[i]
                # Standalone "אבו" followed by complement = 2 tokens consumed
                if tok == 'אבו' and i + 1 < len(nick_tokens):
                    i += 2
                    continue
                # Hyphenated "אבו-X" = 1 token consumed
                if tok.startswith('אבו-'):
                    i += 1
                    continue
                # Non-kunya token found
                non_kunya_count += 1
                i += 1
            is_kunya_only = (non_kunya_count == 0) if nick_tokens else False

            if not is_kunya_only:
                # A valid (non-kunya-only) nickname upgraded quality - use it as canonical
                canonical_to_use = promoted_nick
                # Demote the matched name to nickname set (if different from canonical)
                if cube2_match.name != canonical_to_use:
                    cluster.nicknames.add(cube2_match.name)

        cluster.canonical_name = canonical_to_use
        # Set display_name to raw/original text for output
        display_to_use = cube2_match.name  # raw from cube2
        if contact and contact.get('best_nickname_for_quality') == canonical_to_use:
            # Promoted nickname — use raw version for display
            display_to_use = contact.get('best_nickname_for_quality_raw') or canonical_to_use
        cluster.display_name = display_to_use
        cluster.resolution_type = 'PHONEBOOK'
        # v9: retain cube2 metadata for safe downstream use (bridging/anchors)
        cluster.cube2_contact_key = getattr(cube2_match, 'contact_key', None)
        cluster.phonebook_quality = (getattr(cube2_match, 'quality_tier', '') or 'LOW').upper()
        cluster.phonebook_quality_score = float(getattr(cube2_match, 'quality_score', 0.0) or 0.0)
        if cluster.phonebook_quality != 'HIGH' and 'phonebook_weak' not in cluster.flags:
            cluster.flags.append('phonebook_weak')

        cluster.best_score = cube2_match.score
        cluster.second_best_score = cube2_match.second_score
        cluster.score_margin = cube2_match.margin

        if cube2_match.margin >= 0.25:
            cluster.confidence = 'HIGH'
        elif cube2_match.margin >= self.config.CUBE2_MARGIN_THRESHOLD:
            cluster.confidence = 'MED'
        else:
            cluster.confidence = 'LOW'

        # Add the matched nickname (if not already used as canonical)
        if cube2_match.nickname and cube2_match.nickname != canonical_to_use:
            cluster.nicknames.add(cube2_match.nickname)

        # Propagate full nickname set for high-quality contacts
        # Fix 44: Reuse lookup_phone for cross-phone contact lookups
        if cube2_match.quality_tier == 'HIGH' or cube2_match.entity_id:
            contact = self.cube2_matcher.get_contact_by_key(lookup_phone, cube2_match.contact_key)
            if contact and contact.get('nicknames_normalized'):
                cluster.nicknames.update(contact['nicknames_normalized'])

        # Set entity_id: prefer mention-level (cube1) over phonebook (cube2)
        mention_entity_id = None
        mention_status = None
        mention_id_number = None
        for m in cluster.mentions:
            if m.verified_entity_id and (not getattr(m, 'is_blmz', False)):
                mention_entity_id = m.verified_entity_id
                mention_status = m.verified_status
                mention_id_number = m.verified_id_number
                break
        if mention_entity_id:
            cluster.verified_entity_id = mention_entity_id
            cluster.verified_status = mention_status
            cluster.verified_id_number = mention_id_number
        elif cube2_match.entity_id:
            # Fix 16.1 + Fix 16.4: Nickname Safety for Entity ID Propagation
            # If the match was via nickname (not primary name), do NOT propagate entity_id.
            # Nicknames in phonebooks are noisy (e.g., "Taxi" with nickname "Yossi" because
            # the driver's name is Yossi). Propagating entity_id from nickname matches
            # causes unrelated people to merge via a single shared nickname.
            #
            # EXCEPTION (Fix 16.4): If the nickname that matched is the same as the
            # best_nickname_for_quality that was promoted to canonical (Fix 15.1), then
            # allow entity_id propagation. This handles the case where "Mom" has nickname
            # "Rivka Cohen" (HIGH quality) - when we match on "Rivka Cohen" and promote it
            # to canonical, the entity_id should propagate because "Rivka Cohen" is a
            # high-quality identity, not a noisy nickname like "Yossi" on "Taxi".
            nickname_is_promoted_canonical = (
                cube2_match.nickname and
                contact and
                contact.get('best_nickname_for_quality') == cube2_match.nickname
            )

            # Fix 28.5 (Finding C): Family Corroboration Exception for Nickname Matches
            # When matched via nickname, entity_id is blocked even if family name corroborates.
            # Example: Cluster "אבו-אחמד חג׳אג׳" vs Phonebook name="מחמד חג׳אג׳", nickname="אבו-אחמד"
            # The family "חג׳אג׳" matches perfectly, providing strong corroborating evidence.
            # In this case, allow entity_id propagation because we have identity confirmation
            # beyond just the nickname match.
            family_corroborates = False
            if cube2_match.nickname and contact and not nickname_is_promoted_canonical:
                cluster_family = None
                for m in cluster.mentions:
                    if m.tokens and len(m.tokens) >= 2:
                        cluster_family = self._extract_family_token(m.tokens)
                        if cluster_family:
                            break

                # Get family token from phonebook contact
                contact_tokens = contact.get('tokens') or []
                contact_family = self._extract_family_token(contact_tokens)

                # Check if family tokens match (fuzzy to handle spelling variants)
                if cluster_family and contact_family:
                    family_match_score = _char_ratio(cluster_family, contact_family)
                    family_corroborates = family_match_score >= 85

            # Fix 29.8: High-score corroboration — score ≥0.92 is strong identity evidence
            high_score_corroborates = cube2_match.score >= 0.92
            if not cube2_match.nickname or nickname_is_promoted_canonical or family_corroborates or high_score_corroborates:
                cluster.verified_entity_id = cube2_match.entity_id
                if not cluster.verified_status:
                    cluster.verified_status = cube2_match.status
                if not cluster.verified_id_number:
                    cluster.verified_id_number = cube2_match.id_number

        cluster.match_evidence = f"phonebook:{cube2_match.name}(score={cube2_match.score:.2f},margin={cube2_match.margin:.2f},q={cluster.phonebook_quality})"
        return cluster

    def _resolve_as_inferred(self, cluster: EntityCluster, cube2_match=None) -> EntityCluster:
        canonical = self._select_canonical(cluster.mentions)
        cluster.canonical_name = canonical
        # Find raw version of the selected canonical mention
        display = canonical  # fallback to normalized
        for m in cluster.mentions:
            if m.normalized == canonical:
                display = m.raw_text
                break
        cluster.display_name = display
        cluster.resolution_type = 'INFERRED'
        cluster.confidence = 'LOW'

        # Extract verified_entity_id from mentions (if any mention has it)
        for m in cluster.mentions:
            if m.verified_entity_id and (not getattr(m, 'is_blmz', False)):
                cluster.verified_entity_id = m.verified_entity_id
                cluster.verified_status = m.verified_status
                cluster.verified_id_number = m.verified_id_number
                break

        if cube2_match:
            cluster.best_score = cube2_match.score
            cluster.second_best_score = cube2_match.second_score
            cluster.score_margin = cube2_match.margin
            cluster.cube2_contact_key = getattr(cube2_match, 'contact_key', None)
            cluster.cube2_candidates = f"{cube2_match.name}({cube2_match.score:.2f},m={cube2_match.margin:.2f},q={getattr(cube2_match, 'quality_tier', '')})"
            # FIX: Only capture entity_id from cube2 when the match is confident AND HIGH quality.
            # (Non-confident cube2 matches are useful as weak evidence, but should NOT become a hard ID.)
            # CRITICAL: LOW/MED quality contacts like "Mom", "Taxi", "David Work" must NOT inject
            # their entity_id, as this would cause unrelated people to merge via generic contact names.
            # Only HIGH quality (all tokens non-generic) is safe for entity_id propagation.
            quality_tier = (getattr(cube2_match, 'quality_tier', '') or '').upper()

            # Fix 16.2: Apply IDF demotion and nickname safety to INFERRED path (consistency with PHONEBOOK)
            # Previously, this path didn't check IDF or nickname, allowing "David Cohen" (low IDF) or
            # nickname matches to inject entity_id unsafely.
            effective_tier = quality_tier
            # Fix 19.1: Initialize contact before conditional to avoid NameError
            contact = None
            # Fix 44: Use source_phone for cross-phone contact lookups
            # When match_global() found contact on Phone B but cluster is on Phone A,
            # we must use the contact's source phone to retrieve full contact details.
            lookup_phone = getattr(cube2_match, 'source_phone', None) or cluster.phone
            if cube2_match.entity_id and lookup_phone and cube2_match.contact_key:
                contact = self.cube2_matcher.get_contact_by_key(lookup_phone, cube2_match.contact_key)
                if contact:
                    # Fix 29.22: Default to 5.0 to match finalize_contact_quality().
                    mean_idf = contact.get('mean_idf', 5.0)
                    # Fix 21.1 + Fix 29.10: IDF demotion with configurable threshold
                    if mean_idf < self.config.IDF_DEMOTION_THRESHOLD and effective_tier == 'HIGH':
                        effective_tier = 'MED'

            # Apply nickname safety (Fix 16.1 consistency) with promoted-canonical exception (Fix 16.4)
            nickname_safe = not cube2_match.nickname  # Only safe if matched on primary name
            # Fix 16.4: If nickname is the promoted canonical (best_nickname_for_quality), it's trusted
            if cube2_match.nickname and contact and contact.get('best_nickname_for_quality') == cube2_match.nickname:
                nickname_safe = True

            # Fix 28.5 (Finding C): Family Corroboration Exception for Nickname Matches
            # Same logic as in _resolve_as_phonebook, with Fix 27.3 kunya-at-end handling
            if cube2_match.nickname and contact and not nickname_safe:
                cluster_family = None
                for m in cluster.mentions:
                    if m.tokens and len(m.tokens) >= 2:
                        cluster_family = self._extract_family_token(m.tokens)
                        if cluster_family:
                            break

                contact_tokens = contact.get('tokens') or []
                contact_family = self._extract_family_token(contact_tokens)

                if cluster_family and contact_family:
                    family_match_score = _char_ratio(cluster_family, contact_family)
                    if family_match_score >= 85:
                        nickname_safe = True

            # Fix 28.6 (Finding G): Allow MED tier with entity_id exception (consistency with PHONEBOOK)
            # Previously required effective_tier == 'HIGH', but PHONEBOOK allows MED+entity_id.
            tier_allows_entity_id = (
                effective_tier == 'HIGH' or
                (effective_tier == 'MED' and cube2_match.entity_id)
            )

            if (self._cube2_is_confident(cube2_match)
                and (not cluster.verified_entity_id)
                and cube2_match.entity_id
                and tier_allows_entity_id
                and (nickname_safe or cube2_match.score >= 0.92)):
                cluster.verified_entity_id = cube2_match.entity_id
                if not cluster.verified_status:
                    cluster.verified_status = cube2_match.status
                if not cluster.verified_id_number:
                    cluster.verified_id_number = cube2_match.id_number

            # Fix 21.3: Propagate nicknames from cube2 contact (consistency with PHONEBOOK path)
            # Previously, INFERRED path never populated cluster.nicknames from cube2_match
            if cube2_match.quality_tier == 'HIGH' or cube2_match.entity_id:
                if contact and contact.get('nicknames_normalized'):
                    cluster.nicknames.update(contact['nicknames_normalized'])
                # Also add the matched nickname if different from canonical
                if cube2_match.nickname and cube2_match.nickname != cluster.canonical_name:
                    cluster.nicknames.add(cube2_match.nickname)

        cluster.match_evidence = "inferred_from_cluster"
        return cluster

    def _resolve_as_blmz(self, cluster: EntityCluster) -> EntityCluster:
        cluster.canonical_name = 'בלמ״ז'
        cluster.display_name = 'בלמ״ז'
        cluster.resolution_type = 'BLMZ'
        cluster.confidence = 'LOW'
        cluster.flags.append('unknown_speaker')
        cluster.match_evidence = "blmz_marker"
        return cluster

    def _select_canonical(self, mentions: List[NameMention]) -> str:
        """Select the most representative mention as canonical name."""
        if not mentions:
            return ''

        best_score = -float('inf')
        best_name = mentions[0].normalized if mentions else ''

        for m in mentions:
            if not m.normalized:
                continue

            score = 0
            token_count = len(m.tokens)
            char_count = len(m.normalized)

            if 2 <= token_count <= 4:
                score += 10
            elif token_count == 1:
                score -= 5
            elif token_count > 5:
                score -= 3

            if 5 <= char_count <= 30:
                score += 5
            elif char_count < 3:
                score -= 10

            noise_tokens = self.config.get_noise_tokens_normalized()
            for token in m.tokens:
                if token in noise_tokens:
                    score -= 2

            # Fix 21.5: Deterministic tie-breaking when scores are equal
            # Use lexicographically smallest name to ensure consistent canonical selection
            # regardless of mention processing order
            if score > best_score or (score == best_score and m.normalized < best_name):
                best_score = score
                best_name = m.normalized

        return best_name

    # =========================================================================
    # V8 ARCHITECTURE: Global Cluster Merge with Constraints
    # =========================================================================

    def build_cluster_signature(self, cluster: EntityCluster) -> 'IdentitySignature':
        """Build identity signature for a cluster with all name variants.

        Reuses the variant-generation logic from find_cross_phone_links:
        - Kunya prefix stripping (אבו only)
        - Kunya suffix stripping
        - Connector removal (ב״ר, בר, בן)
        - אל-prefix stripping
        """
        normalizer = self.normalizer
        noise = self.noise_tokens

        def _norm(s: Optional[str]) -> str:
            return normalizer.normalize(s) if s else ""

        def _tokens(name: str) -> List[str]:
            return [t for t in normalizer.tokenize(name) if t and t not in noise]

        def _api_norm(s):
            return normalize_for_api(s) if s else ""

        is_inferred = (cluster.resolution_type == 'INFERRED')
        api_canonical = "" if is_inferred else _api_norm(cluster.canonical_name)

        all_names_api: Set[str] = set()
        if api_canonical:
            all_names_api.add(api_canonical)

        verified_names: Set[str] = set()
        verified_nicks: Set[str] = set()
        mention_counts: Counter = Counter()

        for m in cluster.mentions:
            if m.verified_name:
                verified_names.add(_norm(m.verified_name))
                v = _api_norm(m.verified_name)
                if v:
                    all_names_api.add(v)
            if m.verified_nicknames:
                for nick in _split_nicknames(m.verified_nicknames):
                    verified_nicks.add(_norm(nick))
            if m.normalized:
                mention_counts[m.normalized] += 1
            if m.raw_text:
                v = _api_norm(m.raw_text)
                if v:
                    all_names_api.add(v)
                    if is_inferred and not api_canonical and m.normalized == cluster.canonical_name:
                        api_canonical = v

        # INFERRED fallback if no mention matched canonical
        if is_inferred and not api_canonical:
            api_canonical = _api_norm(cluster.canonical_name)
            if api_canonical:
                all_names_api.add(api_canonical)

        for nick in (cluster.nicknames or []):
            if nick:
                v = _api_norm(nick)
                if v:
                    all_names_api.add(v)

        top_mentions: Set[str] = set([n for n, _ in mention_counts.most_common(5)])
        nicknames: Set[str] = set([_norm(n) for n in (cluster.nicknames or []) if n])
        canonical = _norm(cluster.canonical_name) if cluster.canonical_name else ""

        # Fix 29.12: Filter kunya-only nicknames from all_names to prevent false
        # cross-phone bridges via common kunyas like "אבו-אחמד"
        def _is_kunya_only_name(name: str) -> bool:
            toks = name.split()
            if not toks:
                return False
            i = 0
            while i < len(toks):
                t = toks[i]
                if t == 'אבו' and i + 1 < len(toks):
                    i += 2  # standalone אבו + complement = kunya pair
                    continue
                if t.startswith('אבו-'):
                    i += 1  # hyphenated אבו-X = single kunya token
                    continue
                return False  # non-kunya token found
                i += 1
            return True

        all_names: Set[str] = set()
        if canonical:
            all_names.add(canonical)
        all_names |= {n for n in nicknames if not _is_kunya_only_name(n)}
        all_names |= verified_names
        # Fix 30.0: Filter kunya-only verified nicknames from all_names_normalized,
        # consistent with Fix 29.12. They remain in verified_nicks and all_names_api.
        all_names |= {n for n in verified_nicks if not _is_kunya_only_name(n)}
        all_names |= top_mentions

        # Kunya prefix stripping - only אבו is treated as kunya
        kunya_prefixes = ('אבו ',)

        # Kunya phrase drop for longer names (4+ tokens)
        kunya_phrase_prefixes = ('אבו ',)
        for name in list(all_names):
            toks = name.split()
            if len(toks) >= 4:
                for prefix in kunya_phrase_prefixes:
                    if name.startswith(prefix):
                        core = ' '.join(toks[2:])
                        if core and len(core.split()) >= 2:
                            all_names.add(core)
                        break

        # Fix 26.4: Handle HYPHENATED kunya like "אבו-אחמד אחג׳ד ג׳נליה"
        # When first token is "אבו-X", strip it to get "אחג׳ד ג׳נליה" as a variant
        for name in list(all_names):
            toks = name.split()
            if len(toks) >= 2 and toks[0].startswith('אבו-'):
                # First token is hyphenated kunya, strip it
                core = ' '.join(toks[1:])
                if core and len(core.split()) >= 2:
                    all_names.add(core)

        # Connector removal (ב״ר, בר, בן)
        connector_tokens = {'ב״ר', 'בר', 'בן', "ב''ר", 'ב"ר'}
        for name in list(all_names):
            toks = name.split()
            cleaned = [t for t in toks if t not in connector_tokens]
            if len(cleaned) >= 2 and len(cleaned) < len(toks):
                all_names.add(' '.join(cleaned))

        # Suffix kunya drop - only אבו is treated as kunya
        suffix_kunya_heads = {'אבו'}
        for name in list(all_names):
            toks = name.split()
            if len(toks) >= 4:
                if toks[-2] in suffix_kunya_heads:
                    core = ' '.join(toks[:-2])
                    if core and len(core.split()) >= 2:
                        all_names.add(core)
                if len(toks) >= 5 and toks[-3] in suffix_kunya_heads and toks[-2] == 'אל':
                    core = ' '.join(toks[:-3])
                    if core and len(core.split()) >= 2:
                        all_names.add(core)

        # Kunya prefix stripping (both verified and all names)
        for name in list(verified_nicks):
            for prefix in kunya_prefixes:
                if name.startswith(prefix):
                    stripped = name[len(prefix):]
                    if stripped:
                        all_names.add(stripped)

        for name in list(all_names):
            for prefix in kunya_prefixes:
                if name.startswith(prefix):
                    stripped = name[len(prefix):]
                    if stripped and len(stripped.split()) >= 2:
                        all_names.add(stripped)

        # אל-prefix stripping
        # Generate TWO variants when a token has אל+ו:
        #   1. Al-stripped only (keep ו) — handles names starting with Waw
        #      e.g., "אלואטיה" → "ואטיה" (Watiya / واطية)
        #   2. Al-stripped + ו-stripped — handles conjunction ו
        #      e.g., "אלואטיה" → "אטיה"
        # Previously only variant 2 was generated, losing the correct
        # intermediate form for names like ואטיה, ואליד, וסים, ואא'ל.
        al_strip = re.compile(r'^אל')
        for name in list(all_names):
            toks = _tokens(name)
            if not toks:
                continue
            # Pass 1: al-strip only (keep ו)
            al_only_toks = []
            has_waw = False
            for tok in toks:
                tok_clean = al_strip.sub('', tok)
                if tok_clean != tok and tok_clean.startswith('ו') and len(tok_clean) > 2:
                    has_waw = True
                al_only_toks.append(tok_clean)
            al_only_name = ' '.join(al_only_toks)
            if al_only_name and al_only_name != name:
                all_names.add(al_only_name)
            # Pass 2: al-strip + ו-strip (existing behavior)
            if has_waw:
                al_waw_toks = []
                for tok in toks:
                    tok_clean = al_strip.sub('', tok)
                    if tok_clean.startswith('ו') and len(tok_clean) > 2:
                        tok_clean = tok_clean[1:]
                    al_waw_toks.append(tok_clean)
                al_waw_name = ' '.join(al_waw_toks)
                if al_waw_name and al_waw_name != name:
                    all_names.add(al_waw_name)

        # Build blocking keys
        # Fix: Apply phonetic normalization to first/last tokens for consistent matching
        # This ensures "קאסם" and "כאסם" match during overlap checks, not just fuzzy comparison
        first_tokens: Set[str] = set()
        last_tokens: Set[str] = set()
        max_len = 0

        for name in all_names:
            toks = _tokens(name)
            if not toks:
                continue
            max_len = max(max_len, len(toks))
            if len(toks) >= 2:
                first_tokens.add(normalize_arabic_phonetic(toks[0]))
                last_tokens.add(normalize_arabic_phonetic(toks[-1]))

        return IdentitySignature(
            cluster_id=cluster.cluster_id,
            phone=cluster.phone,
            resolution_type=cluster.resolution_type,
            verified_entity_id=cluster.verified_entity_id,
            phonebook_quality=getattr(cluster, 'phonebook_quality', ''),
            verified_names=verified_names,
            verified_nicknames=verified_nicks,
            canonical_name=canonical,
            nicknames=nicknames,
            top_mentions=top_mentions,
            all_names_normalized=all_names,
            api_name=api_canonical,
            all_names_api=all_names_api,
            first_tokens=first_tokens,
            last_tokens=last_tokens,
            max_token_len=max_len,
        )

    def build_evidence_bundle(self, sig: 'IdentitySignature') -> 'EvidenceBundle':
        """Build an EvidenceBundle from an IdentitySignature for Yanis API scoring.

        Fix 30.1/30.2: Separates strong (identity-defining) names from weak
        (nicknames, kunyas, stripped variants) for principled API aggregation.

        Strong: verified_name (api-normalized), canonical when CALL_VERIFIED/PHONEBOOK HIGH,
                entity_id-backed phonebook main name, multi-token (2+ non-kunya, non-generic) mentions.
        Weak: verified_nicknames, phonebook nicknames, kunya-only forms, single-token names,
              stripped variants.
        """
        # Nested kunya check reuses the same logic as build_cluster_signature
        def _is_kunya_only(name: str) -> bool:
            toks = name.split()
            if not toks:
                return False
            i = 0
            while i < len(toks):
                t = toks[i]
                if t == 'אבו' and i + 1 < len(toks):
                    i += 2  # standalone אבו + complement = kunya pair
                    continue
                if t.startswith('אבו-'):
                    i += 1  # hyphenated אבו-X = single kunya token
                    continue
                return False  # non-kunya token found
                i += 1
            return True

        generic_tokens = self.config.CUBE2_GENERIC_TOKENS

        def _is_strong_name(name: str) -> bool:
            """A name is strong if it has 2+ non-kunya, non-generic tokens."""
            if not name or not name.strip():
                return False
            toks = name.split()
            non_kunya_non_generic = [
                t for t in toks
                if t != 'אבו' and not t.startswith('אבו-') and t not in generic_tokens
            ]
            return len(non_kunya_non_generic) >= 2

        strong: List[str] = []
        weak: List[str] = []
        seen = set()

        for name in sig.all_names_api:
            if not name or not name.strip() or name in seen:
                continue
            seen.add(name)
            if _is_kunya_only(name):
                weak.append(name)
            elif _is_strong_name(name):
                strong.append(name)
            else:
                weak.append(name)

        # Promote api_name / canonical for strong anchor types
        if sig.api_name and sig.api_name not in seen:
            seen.add(sig.api_name)
            if _is_strong_name(sig.api_name):
                strong.append(sig.api_name)
            else:
                weak.append(sig.api_name)

        # Sort strong names longest-first (more tokens = more discriminating)
        strong.sort(key=lambda n: -len(n.split()))
        weak.sort(key=lambda n: -len(n.split()))

        all_api = list(dict.fromkeys(strong + weak))  # deduped, strong first

        return EvidenceBundle(
            cluster_id=sig.cluster_id,
            phone=sig.phone,
            strong_names_api=strong,
            weak_names_api=weak,
            all_names_api=all_api,
            has_verified_name=bool(sig.verified_names),
            has_entity_id=bool(sig.verified_entity_id),
            resolution_type=sig.resolution_type,
            anchor_level=sig.anchor_level(),
        )

    def variant_aware_cluster_score(
        self,
        sig1: 'IdentitySignature',
        sig2: 'IdentitySignature',
        ambiguity_gate: 'AmbiguityGate' = None
    ) -> Tuple[float, str]:
        """Compute similarity between two clusters using all name variants.

        Returns (score, match_type) where match_type describes why they matched.

        Fix 26.9: Accept ambiguity_gate to check global singleton ambiguity.
        """
        best_sort = 0
        best_set = 0
        best_pair = ("", "")

        # v8 Fix B: Allow safe singleton participation
        # Singletons are allowed when:
        # - They match a last_token in the other signature (family name match)
        # - They are rare (>= 5 chars and not a common Hebrew name)
        # Common Arabic names that should NOT act as singleton bridges (too ambiguous)
        # Fix 10.1: Added מחמוד (Mahmoud), עבדאללה (Abdullah) - extremely common names
        # that pass the len>=5 check but should still be blocked.
        COMMON_SINGLETONS = {
            # Common Arabic first names (in Hebrew script)
            'מוחמד', 'מחמד', 'אחמד', 'עלי', 'חסן', 'חוסין', 'יוסף', 'אברהים',
            'מוסטפא', 'עומר', 'סאמי', 'כאמל', 'פאדי', 'ראמי', 'נאסר', 'חאלד',
            'עבד', 'סלים', 'ג׳מאל', 'פריד', 'סעיד', 'מאהר', 'באסם', 'עאדל',
            # Fix 10.1: Common 5+ char names that bypass length check
            'מחמוד', 'עבדאללה', 'איברהים', 'עבדאלרחמן', 'עבדאלכרים',
            # Common Arabic surnames/family indicators
            'אבו', 'אבן', 'אל',
        }
        # Fix 29.39: Pre-compute phonetically normalized version of COMMON_SINGLETONS
        # This ensures phonetic variants (ע→א, ק→כ, ט→ת) are correctly blocked.
        # Example: 'עלי' (Ali with Ayin) normalizes to 'אלי' - both should be blocked.
        COMMON_SINGLETONS_NORMALIZED = {
            normalize_arabic_phonetic(name) for name in COMMON_SINGLETONS
        }
        # Fix 31.7: Also use the broader COMMON_ARABIC_GIVEN_NAMES set.
        # COMMON_SINGLETONS is a manually curated subset that misses names like
        # 'אברהם' (Abraham, 5 chars), 'מוחמוד' (Mahmoud variant, 6 chars) which
        # pass the length>=5 fallback. The Config set covers more common names.
        COMMON_GIVEN_NORMALIZED = self.config.get_common_given_names_normalized()

        def is_safe_singleton(singleton_token: str, other_sig: 'IdentitySignature') -> bool:
            """Check if singleton can safely participate in scoring.

            Fix 26.9: Replace dangerous length heuristic with global ambiguity check.
            The length heuristic (>= 5 chars) was allowing common names like "Israel"
            (ישראל, 5 chars) to act as bridges, causing catastrophic false merges.
            """
            if len(singleton_token) < 2:
                return False  # Too short to be meaningful

            # Fix 31.6: Block kunya singletons from cross-phone bridging.
            if singleton_token.startswith('אבו-'):
                return False

            singleton_phonetic = normalize_arabic_phonetic(singleton_token)

            # Fix 32.4: Check common names BEFORE last_tokens early return.
            # Previously, last_tokens (line 6222) returned True before common name
            # check could block. This allowed "חאלד" to bridge when "אבו חאלד"
            # existed in the other cluster's names (putting "חאלד" in last_tokens).
            # Common given names are never safe as singleton bridges regardless of
            # whether they appear as last tokens — they're too ambiguous.
            if singleton_phonetic in COMMON_SINGLETONS_NORMALIZED or singleton_phonetic in COMMON_GIVEN_NORMALIZED:
                return False

            # Safe if it matches a last token in the other sig (family name match).
            # Only reached by NON-common names (rare family names, distinctive tokens).
            if singleton_phonetic in other_sig.last_tokens:
                return True

            # Fix 26.9 + 27.5: Check GLOBAL ambiguity via AmbiguityGate.
            # - >1 signatures: proven ambiguous → block
            # - ==1 signature: proven unique → allow
            # - ==0 signatures: unknown → fall back to length check
            if ambiguity_gate:
                all_signatures: Set[str] = set()
                token_key = normalize_arabic_phonetic(singleton_token)
                for phone, token_sigs in ambiguity_gate.phone_token_signatures.items():
                    if token_key in token_sigs:
                        all_signatures.update(token_sigs[token_key])

                if len(all_signatures) > 1:
                    return False  # Globally ambiguous

                if len(all_signatures) == 1:
                    return True  # Proven unique across all phones

            # Fall back to length heuristic when:
            # - No ambiguity_gate available, OR
            # - Token has 0 signatures (never seen with a family name - unknown context)
            # This ensures short common Arabic names like "עלי" (Ali, 3 chars) don't act as bridges
            if len(singleton_token) >= 5:
                return True

            return False

        for n1 in sorted(sig1.all_names_normalized):
            t1 = n1.split()
            skip_n1 = False
            if len(t1) < 2:
                # v8 Fix B: Check if this singleton is safe
                if len(t1) == 1 and is_safe_singleton(t1[0], sig2):
                    pass  # Allow this singleton to participate
                else:
                    skip_n1 = True

            for n2 in sorted(sig2.all_names_normalized):
                t2 = n2.split()
                skip_n2 = False
                if len(t2) < 2:
                    # v8 Fix B: Check if this singleton is safe
                    if len(t2) == 1 and is_safe_singleton(t2[0], sig1):
                        pass  # Allow this singleton to participate
                    else:
                        skip_n2 = True

                # Fix 31.7: Skip if EITHER side is an unsafe singleton.
                # Previously only skipped when BOTH were problematic, allowing
                # common singletons like "אברהם" to score against multi-token names
                # ("אברהם כהן") via containment path → false cross-phone merges.
                # When a cluster has both singleton AND full name, the full name
                # will score in its own iteration — the singleton shouldn't contribute.
                if skip_n1 or skip_n2:
                    continue

                # Fix 20.3: Apply phonetic normalization for Arabic-Hebrew homophones
                # This ensures קאסם vs כאסם get 100% similarity, not ~75%
                n1_phonetic = normalize_arabic_phonetic(n1)
                n2_phonetic = normalize_arabic_phonetic(n2)
                sr = _token_sort_ratio(n1_phonetic, n2_phonetic)
                if sr > best_sort or (sr == best_sort and (n1, n2) < best_pair):
                    best_sort = sr
                    best_pair = (n1, n2)

                tr = _token_set_ratio(n1_phonetic, n2_phonetic)
                if tr > best_set:
                    best_set = tr

        # Determine match type
        first_overlap = bool(sig1.first_tokens & sig2.first_tokens)
        last_overlap = bool(sig1.last_tokens & sig2.last_tokens)

        # Fix 28.2 (Finding A): Use config-based anchor thresholds instead of hardcoded values
        # If either signature is a strong anchor (verified name, entity_id, or HIGH phonebook),
        # use the more lenient anchor threshold; otherwise use the strict unverified threshold.
        is_anchor_pair = (
            sig1.anchor_level() == 'strong' or sig2.anchor_level() == 'strong'
        )
        sort_threshold = (
            self.config.CROSS_PHONE_ANCHOR_SORT_THRESHOLD if is_anchor_pair
            else self.config.CROSS_PHONE_UNVERIFIED_SORT_THRESHOLD
        )
        # For containment, use same anchor logic with containment-specific thresholds
        containment_threshold = (
            self.config.CROSS_PHONE_ANCHOR_CONTAINMENT_SET_THRESHOLD if is_anchor_pair
            else self.config.CROSS_PHONE_UNVERIFIED_CONTAINMENT_SET_THRESHOLD
        )

        if best_sort >= sort_threshold:
            return best_sort / 100.0, f'high_sort{"_anchor" if is_anchor_pair else ""}'
        elif first_overlap and last_overlap:
            return max(best_sort, best_set) / 100.0, 'first_last_match'
        elif best_set >= containment_threshold:
            return best_set / 100.0, 'containment'
        elif first_overlap or last_overlap:
            return max(best_sort, best_set) / 100.0, 'partial_overlap'
        else:
            return max(best_sort, best_set) / 100.0, 'fuzzy'

    # Stage 6.8: Noise detection constants
    HEBREW_NOISE_PHRASES = {
        'שיחת בדיקה', 'בדיקת קו', 'צד ב', 'צד א',
        'מספר לא ידוע', 'נמען לא ידוע', 'שיחות אחרות',
        'שיחה נכנסת', 'שיחה יוצאת', 'שיחות נוספות',
        'הנ״ל', 'הנל', 'כנ״ל', 'כנל', 'קו חדש',
    }

    # ------------------------------------------------------------------
    # Stage 6.2: Verified-Anchor API Attachment (Fix 30.1)
    # ------------------------------------------------------------------
    def merge_clusters_by_verified_api_anchors(
        self,
        clusters: List[EntityCluster],
        api_client: 'YanisAPIClient' = None,
        ambiguity_gate: 'AmbiguityGate' = None,
    ) -> List[EntityCluster]:
        """Stage 6.2: Attach entity_ids to unresolved clusters via Yanis API.

        Compares unresolved clusters against verified anchors using the Yanis
        batch API. When a match is confirmed, injects entity_id directly onto
        the cluster so that Stage 7 Phase 0 can hard-link them (bypassing
        the cohesion gate that would otherwise block mention-level mismatches).

        Fix 30.1: This solves cases where variant_aware_cluster_score finds
        a 1.0 match at signature level but the cohesion gate blocks at
        mention level (e.g., nicknames vs actual names).
        """
        cfg = self.config
        if not api_client or not cfg.API_ANCHOR_ATTACH_ENABLED:
            return clusters

        # 5A: Identify anchors and candidates
        anchors: Dict[str, EntityCluster] = {}       # cid -> cluster
        anchor_sigs: Dict[str, IdentitySignature] = {}
        anchor_bundles: Dict[str, EvidenceBundle] = {}

        candidates: Dict[str, EntityCluster] = {}    # cid -> cluster
        candidate_sigs: Dict[str, IdentitySignature] = {}
        candidate_bundles: Dict[str, EvidenceBundle] = {}

        for c in clusters:
            # Skip BLMZ
            if (getattr(c, 'resolution_type', None) == 'BLMZ' or
                    ('unknown_speaker' in (c.flags or [])) or
                    any(getattr(m, 'is_blmz', False) for m in (c.mentions or []))):
                continue
            if 'POSSIBLE_NOISE' in (c.flags or []):
                continue

            sig = self.build_cluster_signature(c)
            bundle = self.build_evidence_bundle(sig)

            if (c.resolution_type == 'CALL_VERIFIED' or
                    c.verified_entity_id or
                    (c.resolution_type == 'PHONEBOOK' and
                     getattr(c, 'phonebook_quality', '') == 'HIGH')):
                # This is an anchor
                if not c.verified_entity_id:
                    continue  # Anchors without entity_id can't attach anything
                anchors[c.cluster_id] = c
                anchor_sigs[c.cluster_id] = sig
                anchor_bundles[c.cluster_id] = bundle
            elif c.resolution_type == 'INFERRED':
                # Candidate: must have 2+ token mentions and not be kunya-only
                has_multi_token = any(len(m.tokens) >= 2 for m in c.mentions)
                if not has_multi_token:
                    continue
                if self._is_kunya_only_cluster(c.mentions):
                    continue
                if c.verified_entity_id:
                    continue  # Already has entity_id, skip
                candidates[c.cluster_id] = c
                candidate_sigs[c.cluster_id] = sig
                candidate_bundles[c.cluster_id] = bundle

        if not anchors or not candidates:
            return clusters

        # 5B: Shortlisting — for each candidate, find top-K anchors by token overlap
        shortlist: Dict[str, List[str]] = {}  # cid -> list of anchor cids
        max_cands = cfg.API_ANCHOR_ATTACH_MAX_CANDIDATES

        for cand_cid, cand_bundle in candidate_bundles.items():
            cand_tokens = set()
            for name in cand_bundle.all_names_api:
                cand_tokens.update(name.split())
            # Include candidate weak names (nicknames) for overlap matching
            for name in cand_bundle.weak_names_api:
                cand_tokens.update(name.split())

            scored_anchors = []
            for anch_cid, anch_bundle in anchor_bundles.items():
                anch_tokens = set()
                for name in anch_bundle.all_names_api:
                    anch_tokens.update(name.split())
                # Include weak names (nicknames) for overlap
                for name in anch_bundle.weak_names_api:
                    anch_tokens.update(name.split())
                overlap = len(cand_tokens & anch_tokens)
                if overlap > 0:
                    scored_anchors.append((overlap, anch_cid))

            scored_anchors.sort(reverse=True)
            shortlist[cand_cid] = [a_cid for _, a_cid in scored_anchors[:max_cands]]

        # Filter out candidates with no shortlisted anchors
        shortlist = {k: v for k, v in shortlist.items() if v}
        if not shortlist:
            return clusters

        # 5C: Batch API call (asymmetric)
        unresolved_names = set()
        anchor_names = set()
        for cand_cid in shortlist:
            for name in candidate_bundles[cand_cid].strong_names_api:
                if name and name.strip():
                    unresolved_names.add(name)
        for anch_cid in set(a for lst in shortlist.values() for a in lst):
            bundle = anchor_bundles[anch_cid]
            for name in bundle.strong_names_api + bundle.weak_names_api:
                if name and name.strip():
                    anchor_names.add(name)

        if unresolved_names and anchor_names:
            api_client.prefetch_anchor_attach(
                unresolved_names, anchor_names,
                min_score=cfg.API_ANCHOR_ATTACH_STRONG_MIN_SCORE)

        if not api_client._api_success.get('anchor_attach', False):
            return clusters

        # 5D: Score aggregation
        _diag = {'attached': 0, 'vetoed': 0, 'skipped': 0, 'blocked': 0}
        for cand_cid, anch_cids in shortlist.items():
            cand_bundle = candidate_bundles[cand_cid]
            cand_cluster = candidates[cand_cid]

            scored: List[Tuple[float, float, int, float, str]] = []
            for anch_cid in anch_cids:
                anch_bundle = anchor_bundles[anch_cid]
                agg = api_client.best_scores_for_bundles(
                    cand_bundle.strong_names_api, cand_bundle.weak_names_api,
                    anch_bundle.strong_names_api, anch_bundle.weak_names_api)
                scored.append((
                    agg['best_strong_to_strong'],
                    agg['best_strong_to_weak'],
                    agg['support_count_ge90'],
                    agg['best_weak_to_weak'],
                    anch_cid,
                ))

            # Sort by best_strong_to_strong descending
            scored.sort(key=lambda x: -x[0])

            if not scored:
                continue

            best_s2s, best_s2w, support_ge90, best_w2w, best_anch_cid = scored[0]
            best_anchor = anchors[best_anch_cid]
            best_anchor_eid = best_anchor.verified_entity_id

            # 5E: Decision policy
            decision = 'SKIP'

            if best_s2s >= cfg.API_ANCHOR_ATTACH_CONFIRM_THRESHOLD:
                decision = 'ATTACH'
            elif (best_s2s >= cfg.API_ANCHOR_ATTACH_DUAL_CONFIRM and
                  support_ge90 >= cfg.API_ANCHOR_ATTACH_DUAL_MIN_SUPPORT):
                decision = 'ATTACH'
            elif (best_s2w >= cfg.API_ANCHOR_ATTACH_WEAK_CONFIRM and
                  best_anchor.verified_entity_id):
                decision = 'ATTACH'
            elif best_s2s < cfg.API_ANCHOR_ATTACH_VETO_THRESHOLD:
                decision = 'VETO'

            # Fix 31.1: Block attachment when given names are confusable
            if decision == 'ATTACH' and cfg.CONFUSABLE_NAME_GUARD_ENABLED:
                _cand_strong = cand_bundle.strong_names_api
                _anch_strong = anchor_bundles[best_anch_cid].strong_names_api
                _conf, _conf_reason = _names_have_confusable_given(
                    _cand_strong, _anch_strong,
                    cfg.get_common_given_names_normalized(),
                    cfg.CONFUSABLE_GIVEN_NAME_MIN_SIMILARITY,
                    cfg.CONFUSABLE_GIVEN_NAME_MAX_IDENTITY,
                    cfg.CONFUSABLE_FAMILY_MATCH_MIN_SIMILARITY)
                if _conf and best_s2s < cfg.API_ANCHOR_ATTACH_CONFUSABLE_THRESHOLD:
                    decision = 'SKIP'
                    cand_cluster.flags.append(f'confusable_given_name_blocked({_conf_reason})')
                    _diag['skipped'] += 1

            # 5E (cont): Double-attachment check — margin between top-2 different-eid anchors
            if decision == 'ATTACH' and len(scored) >= 2:
                second_s2s = scored[1][0]
                second_anch_cid = scored[1][4]
                second_eid = anchors[second_anch_cid].verified_entity_id
                if (second_eid and best_anchor_eid and
                        second_eid != best_anchor_eid and
                        (best_s2s - second_s2s) < cfg.API_ANCHOR_ATTACH_MARGIN_MIN):
                    decision = 'SKIP'
                    cand_cluster.flags.append('yanis_anchor_ambiguous')
                    _diag['skipped'] += 1

            # 5F: Safety gates (before ATTACH)
            if decision == 'ATTACH':
                # Entity_id conflict
                if (cand_cluster.verified_entity_id and
                        cand_cluster.verified_entity_id != best_anchor_eid):
                    decision = 'SKIP'
                    _diag['blocked'] += 1

                # Must-not-link: scan ALL clusters sharing target entity_id
                if decision == 'ATTACH':
                    eid_clusters = [c for c in clusters
                                    if c.verified_entity_id == best_anchor_eid]
                    blocked_by_mnl = False
                    cand_mention_ids = {m.mention_id for m in cand_cluster.mentions}
                    for ec in eid_clusters:
                        for m in ec.mentions:
                            if m.must_not_link & cand_mention_ids:
                                blocked_by_mnl = True
                                break
                        if blocked_by_mnl:
                            break
                    if blocked_by_mnl:
                        decision = 'SKIP'
                        cand_cluster.flags.append('yanis_anchor_blocked_by_must_not_link')
                        _diag['blocked'] += 1

            # 5G: Attachment mechanics (DIRECT MUTATION — no re-resolve)
            if decision == 'ATTACH':
                cand_cluster.verified_entity_id = best_anchor_eid
                cand_cluster.flags.append('yanis_anchor_attached')
                if cand_cluster.match_evidence:
                    cand_cluster.match_evidence += f';yanis_anchor(api={best_s2s:.0f},anchor={best_anch_cid})'
                else:
                    cand_cluster.match_evidence = f'yanis_anchor(api={best_s2s:.0f},anchor={best_anch_cid})'
                _diag['attached'] += 1
                if api_client:
                    api_client.stats['attached'] += 1
            elif decision == 'VETO':
                _diag['vetoed'] += 1
            else:
                _diag['skipped'] += 1

            # 5H: Diagnostic logging (top 3 candidates)
            for rank, (s2s, s2w, sup, w2w, a_cid) in enumerate(scored[:3]):
                dec = decision if rank == 0 else 'SKIP'
                api_client.decisions.append({
                    'decision': f'anchor_{dec.lower()}',
                    'cid1': cand_cid,
                    'cid2': a_cid,
                    'best_strong_to_strong': s2s,
                    'support_count_ge90': sup,
                    'best_strong_to_weak': s2w,
                    'best_weak_to_weak': w2w,
                    'rank': rank,
                })

        logging.info(f"[YANIS] Stage 6.2: attached={_diag['attached']} "
                     f"vetoed={_diag['vetoed']} skipped={_diag['skipped']} "
                     f"blocked={_diag['blocked']} "
                     f"anchors={len(anchors)} candidates={len(candidates)}")

        return clusters

    def _detect_noise_clusters(self, clusters):
        """Stage 6.8: Flag INFERRED singletons that are not person names."""
        for cluster in clusters:
            if cluster.resolution_type != 'INFERRED':
                continue
            name = cluster.canonical_name.strip()
            is_noise = (
                name in self.HEBREW_NOISE_PHRASES or
                name.replace(' ', '').isdigit() or
                len(name) <= 1
            )
            if is_noise:
                cluster.flags.append('POSSIBLE_NOISE')
        return clusters

    def global_cluster_merge(
        self,
        clusters: List[EntityCluster],
        scorer: 'SimilarityScorer' = None,
        ambiguity_gate: 'AmbiguityGate' = None,
        api_client: 'YanisAPIClient' = None
    ) -> ClusterDSU:
        """Replace Stage 7 with global DSU merge over clusters.

        Key improvements:
        - Uses ClusterDSU with constraint enforcement
        - Enforces entity_id conflicts DURING merge
        - Applies cohesion gate to prevent bridge collapse
        - Variant-aware scoring using all name variants

        Args:
            clusters: List of EntityCluster to merge globally
            scorer: Optional SimilarityScorer for cohesion gate (v8 Fix C)
            ambiguity_gate: Optional AmbiguityGate for global singleton safety (Fix 26.9)
        """
        from itertools import combinations

        if not clusters:
            return ClusterDSU()

        dsu = ClusterDSU()

        # Initialize DSU with all clusters and their constraints
        cluster_by_id: Dict[str, EntityCluster] = {}
        sig_by_id: Dict[str, IdentitySignature] = {}

        # Fix 27.4: Aggregate must_not_link constraints from mention level to cluster level
        # First pass: collect mergeable clusters and build mention_id -> cluster_id mapping
        mention_to_cluster: Dict[str, str] = {}
        mergeable_cluster_ids: Set[str] = set()

        for c in clusters:
            # BLMZ SAFETY: exclude unknown-speaker clusters from global merging.
            # Even if a row carries a line-level verified_entity_id, a BLMZ marker means
            # "this speaker is explicitly NOT identified", so it must never participate
            # in ID hard-linking or cross-phone fusion.
            if (getattr(c, 'resolution_type', None) == 'BLMZ' or
                    ('unknown_speaker' in (c.flags or [])) or
                    (c.canonical_name or '') in ('בלמ״ז', 'בלמז') or
                    any(getattr(m, 'is_blmz', False) for m in (c.mentions or []))):
                continue

            if 'POSSIBLE_NOISE' in (c.flags or []):
                continue

            mergeable_cluster_ids.add(c.cluster_id)
            for m in (c.mentions or []):
                mention_to_cluster[m.mention_id] = c.cluster_id

            cluster_by_id[c.cluster_id] = c
            sig = self.build_cluster_signature(c)

            # Enrich signature with cube2 nicknames for PHONEBOOK-resolved clusters
            if c.resolution_type == 'PHONEBOOK' and c.nicknames:
                for nick in c.nicknames:
                    nick_norm = self.normalizer.normalize(nick) if nick else ''
                    if nick_norm:
                        sig.verified_nicknames.add(nick_norm)
                        # Fix 29.12: Don't add kunya-only nicknames to all_names_normalized
                        # to prevent false cross-phone bridges via common kunyas like "אבו-אחמד"
                        nick_toks = nick_norm.split()
                        is_kunya_only = bool(nick_toks) and all(
                            t == 'אבו' or t.startswith('אבו-')
                            for t in nick_toks
                        )
                        if not is_kunya_only:
                            sig.all_names_normalized.add(nick_norm)
                        api_nick = normalize_for_api(nick)
                        if api_nick:
                            sig.all_names_api.add(api_nick)
                # Also add canonical name as searchable variant
                if c.canonical_name:
                    canon_norm = self.normalizer.normalize(c.canonical_name)
                    if canon_norm:
                        sig.all_names_normalized.add(canon_norm)
                    api_canon = normalize_for_api(c.canonical_name)
                    if api_canon:
                        sig.all_names_api.add(api_canon)

            sig_by_id[c.cluster_id] = sig

        # Fix 30.2: Build evidence bundles for bundle-level API scoring
        bundle_by_id: Dict[str, EvidenceBundle] = {}
        if api_client and self.config.API_BUNDLE_SCORING_ENABLED:
            for cid, sig in sig_by_id.items():
                bundle_by_id[cid] = self.build_evidence_bundle(sig)

        # Fix 27.4: Second pass - compute blocked cluster_ids for each cluster
        # and initialize DSU with both entity_id and blocked constraints
        for cid, c in cluster_by_id.items():
            blocked_cluster_ids: Set[str] = set()
            for m in (c.mentions or []):
                if m.must_not_link:
                    for other_mid in m.must_not_link:
                        other_cid = mention_to_cluster.get(other_mid)
                        # Only block if the other mention is in a different mergeable cluster
                        if other_cid and other_cid != cid and other_cid in mergeable_cluster_ids:
                            blocked_cluster_ids.add(other_cid)

            # Initialize with both entity_id and blocked constraints
            dsu.make_set(cid, entity_id=c.verified_entity_id, blocked=blocked_cluster_ids)

        # Phase 0: Hard-link by verified_entity_id (highest priority)
        by_entity_id: Dict[str, List[str]] = defaultdict(list)
        for cid, sig in sig_by_id.items():
            if sig.verified_entity_id:
                by_entity_id[sig.verified_entity_id].append(cid)

        for eid in sorted(by_entity_id.keys()):
            cluster_ids = sorted(by_entity_id[eid])
            if len(cluster_ids) >= 2:
                first = cluster_ids[0]
                for cid in cluster_ids[1:]:
                    dsu.union(first, cid, reason='ENTITY_ID_HARD_LINK', score=1.0)

        # Phase 1: Score all cluster pairs (small N, so all-pairs is fine)
        edges: List[Tuple[float, str, str, str]] = []  # (score, cid1, cid2, match_type)
        rescue_candidates: List[Tuple[float, str, str, str]] = []  # API rescue candidates

        mergeable_clusters = [cluster_by_id[cid] for cid in sorted(cluster_by_id.keys())]

        for c1, c2 in combinations(mergeable_clusters, 2):
            sig1 = sig_by_id[c1.cluster_id]
            sig2 = sig_by_id[c2.cluster_id]

            # Fix 26.9: Pass ambiguity_gate for global singleton safety check
            score, match_type = self.variant_aware_cluster_score(sig1, sig2, ambiguity_gate)

            # Apply phone boost
            if c1.phone == c2.phone:
                score = min(1.0, score + 0.05)
                match_type = f"{match_type}+same_phone"

            # Only consider edges above threshold
            if score >= 0.70:
                edges.append((score, c1.cluster_id, c2.cluster_id, match_type))
            elif (api_client and self.config.API_RESCUE_ENABLED and
                  self.config.API_RESCUE_SCORE_MIN <= score <= self.config.API_RESCUE_SCORE_MAX):
                if (sig1.max_token_len >= self.config.API_RESCUE_MIN_TOKENS and
                        sig2.max_token_len >= self.config.API_RESCUE_MIN_TOKENS):
                    rescue_candidates.append((score, c1.cluster_id, c2.cluster_id, match_type))

        # Sort by score descending (greedy best-first)
        edges.sort(reverse=True)

        # API Call 1: Veto/Confirm prefetch
        # Fix 30.2: When bundle scoring is enabled, collect ALL strong bundle names
        # instead of single representative name per cluster. Also extend to edges > 0.90
        # so Yanis can be consulted for high-confidence edges too.
        if api_client and self.config.API_VETO_ENABLED:
            veto_names = set()
            use_bundles = self.config.API_BUNDLE_SCORING_ENABLED and bundle_by_id
            for sc, cid1, cid2, mt in edges:
                # Fix 30.2: Extend range to include edges > 0.90 for Yanis consultation
                in_veto_band = self.config.API_VETO_SCORE_MIN <= sc <= self.config.API_VETO_SCORE_MAX
                above_veto_band = sc > self.config.API_VETO_SCORE_MAX
                if in_veto_band or (use_bundles and above_veto_band):
                    if use_bundles:
                        b1 = bundle_by_id.get(cid1)
                        b2 = bundle_by_id.get(cid2)
                        if b1:
                            for name in b1.strong_names_api:
                                veto_names.add(name)
                        if b2:
                            for name in b2.strong_names_api:
                                veto_names.add(name)
                    else:
                        n1 = _api_representative_name(sig_by_id[cid1])
                        n2 = _api_representative_name(sig_by_id[cid2])
                        if n1:
                            veto_names.add(n1)
                        if n2:
                            veto_names.add(n2)
            if veto_names:
                if use_bundles:
                    # Use stage7_bundles prefetch (self-join on all names)
                    bundle_names_by_cid = {}
                    for cid, b in bundle_by_id.items():
                        names_set = set(b.strong_names_api)
                        if names_set:
                            bundle_names_by_cid[cid] = names_set
                    api_client.prefetch_stage7_bundles(
                        bundle_names_by_cid,
                        min_score=int(self.config.API_VETO_THRESHOLD))
                else:
                    api_client.prefetch_veto_confirm(
                        veto_names, min_score=int(self.config.API_VETO_THRESHOLD))

        # Phase 2: Merge with cohesion gate and constraint enforcement
        # Fix 30.2: Bundle-level scoring replaces single-representative scoring
        _diag_phase2 = {'merged': 0, 'skip_dsu': 0, 'skip_veto': 0, 'skip_cohesion': 0, 'cohesion_bypassed': 0}
        use_bundles = self.config.API_BUNDLE_SCORING_ENABLED and bundle_by_id
        # Determine which prefetch batch to check for bundle mode
        _bundle_batch_key = 'stage7_bundles' if use_bundles else 'veto'

        for score, cid1, cid2, match_type in edges:
            # Check if merge is allowed by constraints
            can, why = dsu.can_merge(cid1, cid2)
            if not can:
                _diag_phase2['skip_dsu'] += 1
                continue

            _api_bypass_cohesion = False

            # API veto/confirm — bundle-aware (Fix 30.2)
            # Fix 30.2: Extended to ALL edges (not just 0.70-0.90 band) when bundles enabled
            _api_eligible = False
            if api_client and self.config.API_VETO_ENABLED:
                if use_bundles and api_client._api_success.get(_bundle_batch_key, False):
                    _api_eligible = score >= self.config.API_VETO_SCORE_MIN  # All edges above threshold
                elif api_client._api_success.get('veto', False):
                    _api_eligible = self.config.API_VETO_SCORE_MIN <= score <= self.config.API_VETO_SCORE_MAX

            if _api_eligible:
                if use_bundles:
                    # Fix 30.2: Bundle-level scoring
                    b1 = bundle_by_id.get(cid1)
                    b2 = bundle_by_id.get(cid2)
                    if b1 and b2:
                        agg = api_client.best_scores_for_bundles(
                            b1.strong_names_api, b1.weak_names_api,
                            b2.strong_names_api, b2.weak_names_api)
                        api_score = agg['best_strong_to_strong']
                        if api_score > 0:
                            if api_score < self.config.API_VETO_THRESHOLD:
                                dsu.blocked_merges.append({
                                    'cluster1': cid1, 'cluster2': cid2,
                                    'score': score, 'blocked_by': 'API_VETO_BUNDLE',
                                    'api_score': api_score,
                                    'reason_attempted': match_type,
                                })
                                api_client.stats['vetoed'] += 1
                                api_client.decisions.append({
                                    'decision': 'veto', 'name1': str(b1.strong_names_api[:1]),
                                    'name2': str(b2.strong_names_api[:1]),
                                    'api_score': api_score, 'fuzzy_score': score,
                                    'cid1': cid1, 'cid2': cid2})
                                _diag_phase2['skip_veto'] += 1
                                continue
                            if api_score >= self.config.API_CONFIRM_THRESHOLD:
                                match_type = f"{match_type}_api_confirmed"
                                api_client.stats['confirmed'] += 1
                                api_client.decisions.append({
                                    'decision': 'confirm',
                                    'name1': str(b1.strong_names_api[:1]),
                                    'name2': str(b2.strong_names_api[:1]),
                                    'api_score': api_score, 'fuzzy_score': score,
                                    'cid1': cid1, 'cid2': cid2})
                                if (self.config.API_CONFIRM_BYPASS_COHESION and
                                        api_score >= self.config.API_CONFIRM_BYPASS_THRESHOLD):
                                    # Fix 31.1: Block cohesion bypass for confusable given names
                                    if self.config.CONFUSABLE_NAME_GUARD_ENABLED:
                                        _conf, _conf_reason = _names_have_confusable_given(
                                            b1.strong_names_api, b2.strong_names_api,
                                            self.config.get_common_given_names_normalized(),
                                            self.config.CONFUSABLE_GIVEN_NAME_MIN_SIMILARITY,
                                            self.config.CONFUSABLE_GIVEN_NAME_MAX_IDENTITY,
                                            self.config.CONFUSABLE_FAMILY_MATCH_MIN_SIMILARITY)
                                        if not _conf:
                                            _api_bypass_cohesion = True
                                    else:
                                        _api_bypass_cohesion = True
                else:
                    # Legacy single-representative scoring
                    n1 = _api_representative_name(sig_by_id[cid1])
                    n2 = _api_representative_name(sig_by_id[cid2])
                    if n1 and n2:
                        api_score = api_client.score(n1, n2)
                        if api_score is not None:
                            if api_score < self.config.API_VETO_THRESHOLD:
                                dsu.blocked_merges.append({
                                    'cluster1': cid1, 'cluster2': cid2,
                                    'score': score, 'blocked_by': 'API_VETO',
                                    'api_score': api_score,
                                    'reason_attempted': match_type,
                                })
                                api_client.stats['vetoed'] += 1
                                api_client.decisions.append({
                                    'decision': 'veto', 'name1': n1, 'name2': n2,
                                    'api_score': api_score, 'fuzzy_score': score,
                                    'cid1': cid1, 'cid2': cid2})
                                _diag_phase2['skip_veto'] += 1
                                continue
                            if api_score >= self.config.API_CONFIRM_THRESHOLD:
                                match_type = f"{match_type}_api_confirmed"
                                api_client.stats['confirmed'] += 1
                                api_client.decisions.append({
                                    'decision': 'confirm', 'name1': n1, 'name2': n2,
                                    'api_score': api_score, 'fuzzy_score': score,
                                    'cid1': cid1, 'cid2': cid2})
                                if (self.config.API_CONFIRM_BYPASS_COHESION and
                                        api_score >= self.config.API_CONFIRM_BYPASS_THRESHOLD):
                                    # Fix 31.1: Block cohesion bypass for confusable given names
                                    if self.config.CONFUSABLE_NAME_GUARD_ENABLED:
                                        _sig1_names = list(sig_by_id[cid1].all_names_normalized)
                                        _sig2_names = list(sig_by_id[cid2].all_names_normalized)
                                        _conf, _conf_reason = _names_have_confusable_given(
                                            _sig1_names, _sig2_names,
                                            self.config.get_common_given_names_normalized(),
                                            self.config.CONFUSABLE_GIVEN_NAME_MIN_SIMILARITY,
                                            self.config.CONFUSABLE_GIVEN_NAME_MAX_IDENTITY,
                                            self.config.CONFUSABLE_FAMILY_MATCH_MIN_SIMILARITY)
                                        if not _conf:
                                            _api_bypass_cohesion = True
                                    else:
                                        _api_bypass_cohesion = True
                        elif api_client.was_queried('veto', n1, n2):
                            dsu.blocked_merges.append({
                                'cluster1': cid1, 'cluster2': cid2,
                                'score': score, 'blocked_by': 'API_VETO',
                                'api_score': f'<{int(self.config.API_VETO_THRESHOLD)}',
                                'reason_attempted': match_type,
                            })
                            api_client.stats['vetoed'] += 1
                            api_client.decisions.append({
                                'decision': 'veto', 'name1': n1, 'name2': n2,
                                'api_score': 0, 'fuzzy_score': score,
                                'cid1': cid1, 'cid2': cid2})
                            _diag_phase2['skip_veto'] += 1
                            continue

            # Apply cohesion gate to prevent bridge collapse
            # v8 Fix C: Pass scorer for consistent scoring semantics
            if _api_bypass_cohesion:
                _diag_phase2['cohesion_bypassed'] += 1
            else:
                c1 = cluster_by_id[cid1]
                c2 = cluster_by_id[cid2]
                passes, max_score, mean_score = cohesion_gate_passes(c1, c2, scorer=scorer)

                if not passes:
                    dsu.blocked_merges.append({
                        'cluster1': cid1,
                        'cluster2': cid2,
                        'score': score,
                        'blocked_by': 'COHESION_GATE',
                        'reason_attempted': match_type,
                        'max_score': max_score,
                        'mean_score': mean_score,
                    })
                    _diag_phase2['skip_cohesion'] += 1
                    continue

            # Merge!
            dsu.union(cid1, cid2, reason=match_type, score=score)
            _diag_phase2['merged'] += 1

        # API Call 2: Rescue prefetch
        # Fix 30.2: Use bundle names when bundle scoring enabled
        if api_client and self.config.API_RESCUE_ENABLED and rescue_candidates:
            rescue_names = set()
            for sc, cid1, cid2, mt in rescue_candidates:
                if use_bundles:
                    b1 = bundle_by_id.get(cid1)
                    b2 = bundle_by_id.get(cid2)
                    if b1:
                        rescue_names.update(b1.strong_names_api)
                    if b2:
                        rescue_names.update(b2.strong_names_api)
                else:
                    n1 = _api_representative_name(sig_by_id[cid1])
                    n2 = _api_representative_name(sig_by_id[cid2])
                    if n1:
                        rescue_names.add(n1)
                    if n2:
                        rescue_names.add(n2)
            if rescue_names:
                api_client.prefetch_rescue(
                    rescue_names, min_score=int(self.config.API_RESCUE_THRESHOLD))

        # Phase 3: Rescue post-pass
        # Fix 30.2: Bundle-aware rescue scoring
        if (api_client and self.config.API_RESCUE_ENABLED and
                api_client._api_success.get('rescue', False) and rescue_candidates):
            rescue_candidates.sort(reverse=True)  # strongest fuzzy evidence first
            for orig_score, cid1, cid2, match_type in rescue_candidates:
                # Entity_id invariant
                sig1, sig2 = sig_by_id[cid1], sig_by_id[cid2]
                if not self.config.API_RESCUE_ALLOW_EID_PROPAGATION:
                    eid1 = sig1.verified_entity_id
                    eid2 = sig2.verified_entity_id
                    if (eid1 or eid2) and eid1 != eid2:
                        continue
                # DSU constraints
                can, why = dsu.can_merge(cid1, cid2)
                if not can:
                    continue
                # API score check — bundle or single-name
                api_score = None
                _rescue_n1 = _rescue_n2 = ''
                if use_bundles:
                    b1 = bundle_by_id.get(cid1)
                    b2 = bundle_by_id.get(cid2)
                    if b1 and b2:
                        agg = api_client.best_scores_for_bundles(
                            b1.strong_names_api, b1.weak_names_api,
                            b2.strong_names_api, b2.weak_names_api)
                        api_score = agg['best_strong_to_strong']
                        _rescue_n1 = str(b1.strong_names_api[:1])
                        _rescue_n2 = str(b2.strong_names_api[:1])
                else:
                    n1 = _api_representative_name(sig1)
                    n2 = _api_representative_name(sig2)
                    if n1 and n2:
                        api_score = api_client.score(n1, n2)
                        _rescue_n1, _rescue_n2 = n1, n2
                if api_score is None or api_score < self.config.API_RESCUE_THRESHOLD:
                    continue
                # Fix 31.1: Block rescue for confusable given names
                if self.config.CONFUSABLE_NAME_GUARD_ENABLED:
                    if use_bundles:
                        _rb1 = bundle_by_id.get(cid1)
                        _rb2 = bundle_by_id.get(cid2)
                        _rn1 = _rb1.strong_names_api if _rb1 else []
                        _rn2 = _rb2.strong_names_api if _rb2 else []
                    else:
                        _rn1 = list(sig_by_id[cid1].all_names_normalized)
                        _rn2 = list(sig_by_id[cid2].all_names_normalized)
                    _conf, _conf_reason = _names_have_confusable_given(
                        _rn1, _rn2,
                        self.config.get_common_given_names_normalized(),
                        self.config.CONFUSABLE_GIVEN_NAME_MIN_SIMILARITY,
                        self.config.CONFUSABLE_GIVEN_NAME_MAX_IDENTITY,
                        self.config.CONFUSABLE_FAMILY_MATCH_MIN_SIMILARITY)
                    if _conf:
                        dsu.blocked_merges.append({
                            'cluster1': cid1, 'cluster2': cid2,
                            'score': orig_score,
                            'blocked_by': 'CONFUSABLE_GIVEN_NAME',
                            'reason_attempted': f"API_RESCUE(api={api_score:.0f})",
                            'confusable_reason': _conf_reason,
                        })
                        continue
                # Cohesion gate — bypass for singleton pairs
                c1, c2 = cluster_by_id[cid1], cluster_by_id[cid2]
                both_singletons = len(c1.mentions) == 1 and len(c2.mentions) == 1
                if not (both_singletons and self.config.API_RESCUE_BYPASS_COHESION_SINGLETONS):
                    passes, max_sc, mean_sc = cohesion_gate_passes(c1, c2, scorer=scorer)
                    if not passes:
                        dsu.blocked_merges.append({
                            'cluster1': cid1, 'cluster2': cid2,
                            'score': orig_score, 'blocked_by': 'COHESION_GATE',
                            'reason_attempted': f"API_RESCUE(api={api_score:.0f})",
                        })
                        continue
                # Rescue merge
                dsu.union(cid1, cid2,
                          reason=f"API_RESCUE(api={api_score:.0f},fuzzy={orig_score:.2f})",
                          score=api_score / 100.0)
                api_client.stats['rescued'] += 1
                api_client.decisions.append({
                    'decision': 'rescue', 'name1': _rescue_n1, 'name2': _rescue_n2,
                    'api_score': api_score, 'fuzzy_score': orig_score,
                    'cid1': cid1, 'cid2': cid2})

        # API Call 3: Discovery prefetch — ALL name variants for maximum coverage.
        # The API's ignore_common_first_name=True (line 364) suppresses matches
        # based solely on common first names, preventing false merges from
        # kunya-derived single-token names (e.g. "עלי" from "אבו עלי").
        if api_client and self.config.API_DISCOVERY_ENABLED:
            discovery_names = set()
            for cid, sig in sig_by_id.items():
                for name in sig.all_names_api:
                    if name and name.strip():
                        discovery_names.add(name)
            if discovery_names:
                api_client.prefetch_discovery(
                    discovery_names,
                    min_score=int(self.config.API_DISCOVERY_THRESHOLD))

        # Phase 4: API Discovery — find matches our fuzzy scorer missed
        _diag_phase4 = {
            'entered': False, 'cache_pairs': 0, 'pairs_checked': 0,
            'skip_already_merged': 0, 'skip_both_single_tok': 0,
            'skip_entity_id': 0, 'skip_dsu_blocked': 0,
            'skip_cohesion': 0, 'merged': 0,
        }
        if (api_client and self.config.API_DISCOVERY_ENABLED and
                api_client._api_success.get('discovery', False)):
            _diag_phase4['entered'] = True
            # Map ALL variant names → cluster IDs so cache hits from any
            # API call (1/2/3) can trace back through shared name variants.
            name_to_cids = defaultdict(set)
            for cid, sig in sig_by_id.items():
                for name in sig.all_names_api:
                    if name and name.strip():
                        name_to_cids[name].add(cid)

            discovery_pairs = sorted(
                ((k, v) for k, v in api_client._cache.items()
                 if v >= self.config.API_DISCOVERY_THRESHOLD),
                key=lambda x: -x[1])
            _diag_phase4['cache_pairs'] = len(discovery_pairs)

            for (n1, n2), grade in discovery_pairs:
                cids1 = name_to_cids.get(n1, set())
                cids2 = name_to_cids.get(n2, set())
                for cid1 in cids1:
                    for cid2 in cids2:
                        if cid1 >= cid2:
                            continue
                        _diag_phase4['pairs_checked'] += 1
                        if dsu.find(cid1) == dsu.find(cid2):
                            _diag_phase4['skip_already_merged'] += 1
                            continue  # already in same group
                        # Both-single-token safety: "same name" ≠ "same person"
                        sig1, sig2 = sig_by_id[cid1], sig_by_id[cid2]
                        if sig1.max_token_len < 2 and sig2.max_token_len < 2:
                            _diag_phase4['skip_both_single_tok'] += 1
                            continue
                        # Entity_id check
                        eid1 = sig1.verified_entity_id
                        eid2 = sig2.verified_entity_id
                        if (eid1 or eid2) and eid1 != eid2:
                            _diag_phase4['skip_entity_id'] += 1
                            continue
                        # DSU constraints
                        can, why = dsu.can_merge(cid1, cid2)
                        if not can:
                            _diag_phase4['skip_dsu_blocked'] += 1
                            continue
                        # Cohesion gate (bypass for both-singleton-mention pairs)
                        c1, c2 = cluster_by_id[cid1], cluster_by_id[cid2]
                        both_singletons = (len(c1.mentions) == 1 and
                                           len(c2.mentions) == 1)
                        if not both_singletons:
                            passes, _, _ = cohesion_gate_passes(
                                c1, c2, scorer=scorer)
                            if not passes:
                                dsu.blocked_merges.append({
                                    'cluster1': cid1, 'cluster2': cid2,
                                    'score': grade / 100.0,
                                    'blocked_by': 'COHESION_GATE',
                                    'reason_attempted':
                                        f"API_DISCOVERY(api={grade:.0f})",
                                })
                                _diag_phase4['skip_cohesion'] += 1
                                continue
                        # Fix 31.1: Block discovery merge for confusable given names
                        if self.config.CONFUSABLE_NAME_GUARD_ENABLED:
                            if use_bundles:
                                _db1 = bundle_by_id.get(cid1)
                                _db2 = bundle_by_id.get(cid2)
                                _dn1 = _db1.strong_names_api if _db1 else []
                                _dn2 = _db2.strong_names_api if _db2 else []
                            else:
                                _dn1 = list(sig_by_id[cid1].all_names_normalized)
                                _dn2 = list(sig_by_id[cid2].all_names_normalized)
                            _conf, _conf_reason = _names_have_confusable_given(
                                _dn1, _dn2,
                                self.config.get_common_given_names_normalized(),
                                self.config.CONFUSABLE_GIVEN_NAME_MIN_SIMILARITY,
                                self.config.CONFUSABLE_GIVEN_NAME_MAX_IDENTITY,
                                self.config.CONFUSABLE_FAMILY_MATCH_MIN_SIMILARITY)
                            if _conf:
                                dsu.blocked_merges.append({
                                    'cluster1': cid1, 'cluster2': cid2,
                                    'score': grade / 100.0,
                                    'blocked_by': 'CONFUSABLE_GIVEN_NAME',
                                    'reason_attempted': f"API_DISCOVERY(api={grade:.0f})",
                                    'confusable_reason': _conf_reason,
                                })
                                continue
                        # Discovery merge
                        dsu.union(cid1, cid2,
                                  reason=f"API_DISCOVERY(api={grade:.0f})",
                                  score=grade / 100.0)
                        _diag_phase4['merged'] += 1
                        api_client.stats['discovered'] += 1
                        api_client.decisions.append({
                            'decision': 'discovery',
                            'name1': n1, 'name2': n2,
                            'api_score': grade, 'fuzzy_score': 0.0,
                            'cid1': cid1, 'cid2': cid2})

        # Debug output
        if self.config.DEBUG_CROSS_PHONE:
            groups = dsu.get_groups()
            multi_cluster = {k: v for k, v in groups.items() if len(v) > 1}
            print(f"[GLOBAL_MERGE] Created {len(groups)} global entities, {len(multi_cluster)} span multiple clusters")
            if dsu.blocked_merges:
                print(f"[GLOBAL_MERGE] Blocked {len(dsu.blocked_merges)} merges:")
                for bm in dsu.blocked_merges[:10]:
                    print(f"  - {bm['cluster1']} + {bm['cluster2']}: {bm['blocked_by']} (score={bm['score']:.2f})")
            if api_client:
                print(api_client.log_summary())
                if rescue_candidates:
                    print(f"[RESCUE] candidates={len(rescue_candidates)} "
                          f"rescued={api_client.stats['rescued']}")
                if api_client.stats['discovered']:
                    print(f"[DISCOVERY] discovered={api_client.stats['discovered']}")

        # ── Compact Merge Diagnostic (always printed, screenshot-friendly) ──
        _dg_groups = dsu.get_groups()
        _dg_multi = sum(1 for v in _dg_groups.values() if len(v) > 1)
        _dg_total_clusters = len(sig_by_id)
        _dg_total_pairs = _dg_total_clusters * (_dg_total_clusters - 1) // 2
        _dg_edges = len(edges)
        _dg_rescue = len(rescue_candidates)
        _dg_below = _dg_total_pairs - _dg_edges - _dg_rescue

        print("\n" + "=" * 58)
        print(f" MERGE DIAGNOSTIC  clusters={_dg_total_clusters}"
              f"  pairs_scored={_dg_total_pairs}")
        print("=" * 58)
        print(f" SCORING  edges(\u2265.70)={_dg_edges}"
              f"  rescue(.45-.69)={_dg_rescue}"
              f"  below={_dg_below}")
        d2 = _diag_phase2
        print(f" PHASE2   merged={d2['merged']}"
              f"  skip: dsu={d2['skip_dsu']}"
              f" veto={d2['skip_veto']}"
              f" cohesion={d2['skip_cohesion']}"
              f" api_bypass={d2['cohesion_bypassed']}")
        if not api_client:
            print(f" API      *** NONE ***"
                  f" API_ENABLED={self.config.API_ENABLED}"
                  f" HAS_REQUESTS={HAS_REQUESTS}")
        else:
            s = api_client.stats
            _url_short = api_client.url[:60] + '...' if len(api_client.url) > 60 else api_client.url
            _tok_hint = f"{api_client.token[:12]}..." if len(api_client.token) > 12 else ('(empty)' if not api_client.token else '(set)')
            print(f" API      ACTIVE  calls={s['batch_calls']}"
                  f" cache={len(api_client._cache)}"
                  f" err={s['errors']}"
                  f"  url={_url_short}"
                  f"  token={_tok_hint}")
            if api_client.error_messages:
                for em in api_client.error_messages:
                    print(f"          ERR: {em}")
            print(f"          veto={s['vetoed']}"
                  f" confirm={s['confirmed']}"
                  f" rescued={s['rescued']}"
                  f" discovered={s['discovered']}")
            d4 = _diag_phase4
            if not d4['entered']:
                why = []
                if not self.config.API_DISCOVERY_ENABLED:
                    why.append("DISC_ENABLED=False")
                if not api_client._api_success.get('discovery', False):
                    why.append("disc_call_failed")
                print(f" PHASE4   *** NOT ENTERED *** {', '.join(why)}")
            else:
                print(f" PHASE4   cache\u226575={d4['cache_pairs']}"
                      f" checked={d4['pairs_checked']}"
                      f" \u2192merged={d4['merged']}")
                print(f"          skip: same={d4['skip_already_merged']}"
                      f" 1tok={d4['skip_both_single_tok']}"
                      f" eid={d4['skip_entity_id']}"
                      f" dsu={d4['skip_dsu_blocked']}"
                      f" coh={d4['skip_cohesion']}")
        print(f" RESULT   entities={len(_dg_groups)}"
              f"  multi_phone={_dg_multi}"
              f"  blocked={len(dsu.blocked_merges)}")
        print("=" * 58 + "\n")

        return dsu

    def _head_signature_from_tokens(self, tokens: List[str]) -> str:
        """Compute a lightweight 'head signature' similar to AmbiguityGate logic."""
        if not tokens:
            return ''

        # Prefer a last token that is not an abbreviation/title (gershayim)
        family = None
        for tok in reversed(tokens):
            if '"' not in tok and '״' not in tok:
                family = tok
                break
        family = family or tokens[-1]

        # Arabic kunya / patronymic marker can be signal
        # Fix 22.1: Consistency with AmbiguityGate._get_head_signature - only אבו (Abu) is common
        kunya = None
        for tok in tokens:
            if tok.startswith('אבו'):
                kunya = tok
                break

        return f"{family}+{kunya}" if kunya else family

    def _cube2_bridge_blocked_by_ambiguity(
        self,
        phone: str,
        cluster: EntityCluster,
        cube2_match: Cube2Match,
        ambiguity_gate: AmbiguityGate,
    ) -> bool:
        """Apply a conservative AmbiguityGate-style guard to cube2 bridging.

        If a cluster contains a single-token mention that is ambiguous on this phone
        (i.e., the token maps to multiple head signatures), we require EITHER:
        1. The cube2 contact signature to be one of the observed signatures, OR
        2. The cluster itself to contain a multi-token mention with the same family
           (Fix 29.17: cluster self-disambiguation)

        This prevents blocking valid matches when the cluster contains both a singleton
        AND a full-name mention that disambiguates.
        """
        if not ambiguity_gate:
            return False
        # Fix 32.2: Do NOT bypass ambiguity gate for entity_id contacts.
        # Entity_id contacts are the MOST dangerous (they cascade via Phase 0
        # hard-link), so ambiguity detection must apply to them especially.
        # Previously: "if cube2_match.entity_id: return False" — REMOVED.

        contact_sig = self._head_signature_from_tokens(getattr(cube2_match, 'tokens', []) or [])
        phone_sigs = ambiguity_gate.phone_token_signatures.get(phone, {})

        # Fix 29.17: Collect family signatures from multi-token mentions in the cluster.
        # If any multi-token mention shares the phonebook's family, the cluster
        # provides its own disambiguation evidence.
        cluster_families = set()
        for m in cluster.mentions:
            if len(m.tokens) >= 2:
                # Extract family signature from multi-token mention
                mention_sig = self._head_signature_from_tokens(m.tokens)
                if mention_sig:
                    # Extract just the family part (before any +kunya)
                    family_part = mention_sig.split('+')[0] if '+' in mention_sig else mention_sig
                    if family_part:
                        cluster_families.add(normalize_arabic_phonetic(family_part))

        # Also normalize contact_sig for comparison
        contact_family = None
        if contact_sig:
            contact_family = contact_sig.split('+')[0] if '+' in contact_sig else contact_sig
            contact_family = normalize_arabic_phonetic(contact_family) if contact_family else None

        # If cluster contains the phonebook's family, it self-disambiguates
        if contact_family and contact_family in cluster_families:
            return False  # Allow - cluster provides disambiguation

        for m in cluster.mentions:
            if len(m.tokens) != 1:
                continue
            tok = m.tokens[0] if m.tokens else ''
            if not tok:
                continue
            # Fix 29.1d: Apply phonetic normalization to lookup key to match indexing
            tok_key = normalize_arabic_phonetic(tok)
            sigs = phone_sigs.get(tok_key, set())
            if len(sigs) > 1:
                # Ambiguous token: require cube2 to disambiguate
                if not contact_sig or (contact_sig not in sigs):
                    return True

        return False

    def _cube2_can_merge_clusters(self, c1: EntityCluster, c2: EntityCluster) -> bool:
        """Constraint-only merge check for cube2 bridging (no similarity needed)."""
        # Never merge unresolved clusters
        if c1.cluster_id.endswith('_unresolved') or c2.cluster_id.endswith('_unresolved'):
            return False
        if 'UNRESOLVED_CONFLICT' in c1.flags or 'UNRESOLVED_CONFLICT' in c2.flags:
            return False

        # Never merge BLMZ with non-BLMZ (and keep BLMZ isolated)
        if c1.resolution_type == 'BLMZ' or c2.resolution_type == 'BLMZ':
            return False

        # entity_id conflict guard
        if c1.verified_entity_id and c2.verified_entity_id and c1.verified_entity_id != c2.verified_entity_id:
            return False

        # Verified-name conflict guard (if both have verified evidence and disagree)
        if c1.has_verified() and c2.has_verified():
            if c1.canonical_name != c2.canonical_name:
                return False

        # Respect mention-level must-not-link constraints
        ids2 = c2.mention_ids
        for m in c1.mentions:
            if m.must_not_link and (m.must_not_link & ids2):
                return False
        ids1 = c1.mention_ids
        for m in c2.mentions:
            if m.must_not_link and (m.must_not_link & ids1):
                return False

        return True

    def merge_clusters_by_cube2_anchors(
        self,
        clusters: List[EntityCluster],
        ambiguity_gate: Optional[AmbiguityGate] = None,
    ) -> List[EntityCluster]:
        """Merge clusters within each phone using cube2 as an anchor (Stage 6.25).

        Key properties:
        - Cluster-level (post-HAC) merging: avoids injecting virtual nodes into the
          similarity graph, which is risky for noisy Hebrew phonebooks.
        - Only merges when multiple clusters map to the SAME cube2 contact_key with
          strong margin + score, and subject to verified/entity_id/must-not-link guards.
        - Applies an AmbiguityGate-style check to avoid over-merging ambiguous singletons.
        """
        cfg = self.config
        if not getattr(cfg, 'CUBE2_BRIDGE_ENABLED', True) or not clusters:
            return clusters

        clusters_by_phone: Dict[str, List[EntityCluster]] = defaultdict(list)
        for c in clusters:
            clusters_by_phone[c.phone].append(c)

        merged_total = 0
        out: List[EntityCluster] = []

        for phone, phone_clusters in clusters_by_phone.items():
            if len(phone_clusters) < 2:
                out.extend(phone_clusters)
                continue

            contacts = self.cube2_matcher.contacts_by_phone.get(phone, [])
            if not contacts:
                out.extend(phone_clusters)
                continue

            too_many_contacts = len(contacts) >= int(getattr(cfg, 'CUBE2_BRIDGE_MAX_CONTACTS_PER_PHONE', 25))

            cluster_by_id: Dict[str, EntityCluster] = {c.cluster_id: c for c in phone_clusters}

            # 1) compute eligible cube2 matches per cluster
            match_by_id: Dict[str, Cube2Match] = {}
            for c in phone_clusters:
                if c.resolution_type == 'BLMZ':
                    continue
                if c.cluster_id.endswith('_unresolved') or ('UNRESOLVED_CONFLICT' in c.flags):
                    continue

                m = self.cube2_matcher.match(phone, c.mentions)
                if not m:
                    continue

                if m.score < float(getattr(cfg, 'CUBE2_BRIDGE_MIN_SCORE', 0.75)):
                    continue
                if m.margin < float(getattr(cfg, 'CUBE2_BRIDGE_MIN_MARGIN', 0.20)):
                    continue

                # Quality gating: only HIGH-quality name keys may bridge. If the phone has many
                # contacts, require entity_id (multi-identity risk).
                if not m.entity_id:
                    if (m.quality_tier or '').upper() != 'HIGH':
                        continue
                    if too_many_contacts and bool(getattr(cfg, 'CUBE2_BRIDGE_REQUIRE_ENTITY_ID_WHEN_MANY_CONTACTS', True)):
                        continue

                if ambiguity_gate and self._cube2_bridge_blocked_by_ambiguity(phone, c, m, ambiguity_gate):
                    continue

                match_by_id[c.cluster_id] = m

            if not match_by_id:
                out.extend(phone_clusters)
                continue

            # 2) group clusters by stable cube2 contact key
            key_to_cluster_ids: Dict[str, List[str]] = defaultdict(list)
            for cid, m in match_by_id.items():
                key = m.contact_key or (f"EID:{m.entity_id}" if m.entity_id else None)
                if not key:
                    continue
                key_to_cluster_ids[key].append(cid)

            # DSU init (per phone)
            parent: Dict[str, str] = {cid: cid for cid in cluster_by_id.keys()}

            def _find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def _union(a: str, b: str) -> None:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[rb] = ra

            # 3) attempt merges inside each key group (constraint-only)
            for key, ids in key_to_cluster_ids.items():
                if len(ids) < 2:
                    continue

                has_eid = bool(match_by_id[ids[0]].entity_id)
                max_per = int(getattr(cfg, 'CUBE2_BRIDGE_MAX_CLUSTERS_PER_CONTACT', 8))
                if (not has_eid) and len(ids) > max_per:
                    # Too big a component to merge via name-only key
                    continue

                ids_sorted = sorted(ids)
                for i in range(len(ids_sorted)):
                    for j in range(i + 1, len(ids_sorted)):
                        cid1, cid2 = ids_sorted[i], ids_sorted[j]
                        c1, c2 = cluster_by_id[cid1], cluster_by_id[cid2]
                        if self._cube2_can_merge_clusters(c1, c2):
                            _union(cid1, cid2)

            # 4) materialize components + re-resolve merged clusters
            components: Dict[str, List[str]] = defaultdict(list)
            for cid in cluster_by_id.keys():
                components[_find(cid)].append(cid)

            for root, members in components.items():
                if len(members) == 1:
                    out.append(cluster_by_id[members[0]])
                    continue

                merged_mentions: List[NameMention] = []
                merged_flags: Set[str] = set()
                for cid in members:
                    cc = cluster_by_id[cid]
                    merged_mentions.extend(cc.mentions)
                    merged_flags.update(cc.flags)

                new_cluster_id = min(members)
                new_cluster = self.resolve_cluster(new_cluster_id, phone, merged_mentions)

                # Carry forward flags + annotate
                new_cluster.flags = sorted(set(new_cluster.flags) | merged_flags | {'cube2_bridge_merged'})
                if new_cluster.match_evidence:
                    if 'cube2_bridge' not in new_cluster.match_evidence:
                        new_cluster.match_evidence = new_cluster.match_evidence + '|cube2_bridge'
                else:
                    new_cluster.match_evidence = 'cube2_bridge'

                out.append(new_cluster)
                merged_total += (len(members) - 1)

        if merged_total and getattr(cfg, 'DEBUG_CROSS_PHONE', False):
            print(f"[CUBE2_BRIDGE] merged {merged_total} intra-phone clusters via cube2 anchors")

        return sorted(out, key=lambda c: c.cluster_id)

    def merge_similar_clusters(self, clusters: List[EntityCluster]) -> List[EntityCluster]:
        """Second pass: merge clusters with very similar canonicals (Fix 3.4).

        This catches near-miss pairs that didn't connect at mention level but have very
        similar canonical names.

        Hardening:
        - Deterministic survivor selection: Verified ID > resolution quality > verified-name evidence > score > size.
        - Credential inheritance: verified_entity_id is never dropped during merge.
        - Constraint-aware grouping: never merge a connected component that would contain
          multiple different verified_entity_ids (including transitive A~B, B~C cases).
        - Quarantine stability: clusters marked as unresolved conflicts are never merged here.
        """
        if not clusters:
            return clusters

        # Deterministic processing order
        clusters = sorted(clusters, key=lambda c: c.cluster_id)

        # Group clusters by phone (local merge pass is per-phone only).
        by_phone: Dict[str, List[EntityCluster]] = defaultdict(list)
        for c in clusters:
            by_phone[c.phone].append(c)

        cluster_by_id: Dict[str, EntityCluster] = {c.cluster_id: c for c in clusters}

        # Helper: derive all non-null entity_ids represented by a cluster.
        # We read from both the cluster field and its mentions to avoid stale state.
        def _cluster_entity_ids(c: EntityCluster) -> Set[str]:
            ids: Set[str] = set()
            if c.verified_entity_id:
                ids.add(c.verified_entity_id)
            for m in c.mentions:
                if getattr(m, 'is_blmz', False):
                    continue
                vid = getattr(m, 'verified_entity_id', None)
                if vid:
                    ids.add(vid)
            return ids

        # Constraint-aware DSU for cluster ids
        parent: Dict[str, str] = {}
        comp_ids: Dict[str, Set[str]] = {}

        def _find(x: str) -> str:
            if x not in parent:
                parent[x] = x
                comp_ids[x] = set()
            if parent[x] != x:
                parent[x] = _find(parent[x])
            return parent[x]

        def _union(x: str, y: str) -> bool:
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return True

            merged_ids = comp_ids.get(rx, set()) | comp_ids.get(ry, set())
            # Hard constraint: do NOT merge if it would create conflicting verified IDs.
            if len(merged_ids) > 1:
                return False

            # Deterministic union (by id) to avoid non-deterministic survivors.
            if rx < ry:
                parent[ry] = rx
                comp_ids[rx] = merged_ids
                comp_ids.pop(ry, None)
            else:
                parent[rx] = ry
                comp_ids[ry] = merged_ids
                comp_ids.pop(rx, None)
            return True

        # Initialize DSU and component IDs
        for c in clusters:
            r = _find(c.cluster_id)
            comp_ids[r] = comp_ids.get(r, set()) | _cluster_entity_ids(c)

        # Build connectivity (candidate merges) per phone
        for phone, phone_clusters in by_phone.items():
            for i in range(len(phone_clusters)):
                c1 = phone_clusters[i]
                for j in range(i + 1, len(phone_clusters)):
                    c2 = phone_clusters[j]
                    if self._should_merge_clusters(c1, c2):
                        _union(c1.cluster_id, c2.cluster_id)

        # Materialize DSU groups
        groups: Dict[str, List[str]] = defaultdict(list)
        for cid in parent.keys():
            groups[_find(cid)].append(cid)

        merged_clusters: List[EntityCluster] = []

        # Rank survivor: Verified ID > resolution type > verified name evidence > scores > size > id
        res_pri = {
            'CALL_VERIFIED': 4,
            'PHONEBOOK': 3,
            'INFERRED': 2,
            'BLMZ': 1,
            'VERIFIED_CONFLICT': 0,
        }

        def _rank(c: EntityCluster) -> Tuple[int, int, int, float, float, int, str]:
            return (
                1 if c.verified_entity_id else 0,
                res_pri.get(c.resolution_type or '', 0),
                1 if c.has_verified() else 0,
                float(c.best_score or 0.0),
                float(c.score_margin or 0.0),
                int(c.size),
                c.cluster_id,
            )

        for root, member_ids in groups.items():
            if len(member_ids) == 1:
                merged_clusters.append(cluster_by_id[member_ids[0]])
                continue

            members = [cluster_by_id[mid] for mid in member_ids]

            # If, despite DSU guard, we see conflicts (defensive), refuse merge.
            group_ids: Set[str] = set()
            for c in members:
                group_ids |= _cluster_entity_ids(c)

            if len(group_ids) > 1:
                for c in members:
                    if 'ENTITY_ID_CONFLICT' not in c.flags:
                        c.flags.append('ENTITY_ID_CONFLICT')
                merged_clusters.extend(members)
                continue

            survivor = max(members, key=_rank)

            for victim in members:
                if victim.cluster_id == survivor.cluster_id:
                    continue

                # Merge content
                survivor.mentions.extend(victim.mentions)
                survivor.mention_ids.update(victim.mention_ids)
                survivor.nicknames.update(victim.nicknames)

                # Fix "Zombie Cluster" bug: update cluster_id on moved mentions
                # to prevent dangling references to destroyed/merged clusters
                for m in victim.mentions:
                    m.cluster_id = survivor.cluster_id

                # Merge cross-phone provenance if present
                if getattr(victim, 'cross_phone_links', None):
                    survivor.cross_phone_links.extend(victim.cross_phone_links)

                # Merge flags (unique)
                if victim.flags:
                    existing = set(survivor.flags)
                    for f in victim.flags:
                        if f not in existing:
                            survivor.flags.append(f)

                # Credential inheritance
                if not survivor.verified_entity_id and victim.verified_entity_id:
                    survivor.verified_entity_id = victim.verified_entity_id
                    if not survivor.verified_status:
                        survivor.verified_status = victim.verified_status
                    if not survivor.verified_id_number:
                        survivor.verified_id_number = victim.verified_id_number

            # Recompute survivor entity_id from merged mentions (defensive)
            final_ids = _cluster_entity_ids(survivor)
            if len(final_ids) == 1:
                winner_eid = next(iter(final_ids))
                survivor.verified_entity_id = winner_eid
                # Derive status/id_number from the mention that has this entity_id
                for m in survivor.mentions:
                    if m.verified_entity_id == winner_eid:
                        survivor.verified_status = m.verified_status
                        survivor.verified_id_number = m.verified_id_number
                        break
            elif len(final_ids) > 1:
                survivor.verified_entity_id = None
                survivor.verified_status = None
                survivor.verified_id_number = None
                if 'ENTITY_ID_CONFLICT' not in survivor.flags:
                    survivor.flags.append('ENTITY_ID_CONFLICT')

            if 'CLUSTER_MERGED' not in survivor.flags:
                survivor.flags.append('CLUSTER_MERGED')

            merged_clusters.append(survivor)

        return merged_clusters


    def _should_merge_clusters(self, c1: EntityCluster, c2: EntityCluster) -> bool:
        """Check if two clusters should merge based on canonicals (Fix 3.4).

        Settings are configurable via CLUSTER_MERGE_* thresholds.
        Includes conflict guard to prevent merging different verified people.
        """
        if not c1.canonical_name or not c2.canonical_name:
            return False

        # Guard: BLMZ ("unknown speaker") is a negative constraint.
        # Allow BLMZ↔BLMZ merges (harmless), but never merge BLMZ with a non-BLMZ cluster.
        if (c1.resolution_type == 'BLMZ') != (c2.resolution_type == 'BLMZ'):
            return False

        # Guard: never merge quarantined "unresolved" clusters.
        # These clusters exist precisely because we detected ambiguity; merging them
        # based on canonical similarity re-introduces the original corruption risk.
        if (c1.cluster_id.endswith('_unresolved') or c2.cluster_id.endswith('_unresolved') or
                'UNRESOLVED_CONFLICT' in c1.flags or 'UNRESOLVED_CONFLICT' in c2.flags):
            return False
        # Guard: mention-level must-not-link constraints must be preserved (v9).
        # HAC respected these; post-HAC merges must respect them too.
        ids2 = c2.mention_ids
        for m in c1.mentions:
            if m.must_not_link and (m.must_not_link & ids2):
                return False
        ids1 = c1.mention_ids
        for m in c2.mentions:
            if m.must_not_link and (m.must_not_link & ids1):
                return False


        name1 = self.normalizer.normalize(c1.canonical_name)
        name2 = self.normalizer.normalize(c2.canonical_name)
        # Fix 20.4: Apply phonetic normalization for Arabic-Hebrew homophones
        # This ensures קאסם vs כאסם get proper similarity scores
        name1_phonetic = normalize_arabic_phonetic(name1) if name1 else name1
        name2_phonetic = normalize_arabic_phonetic(name2) if name2 else name2

        if not name1 or not name2:
            return False

        cfg = self.config

        # v8 FIX: ENTITY_ID CONFLICT GUARD
        # Never merge clusters with different verified_entity_ids
        # This is a HARD constraint - entity_id is ground truth
        if c1.verified_entity_id and c2.verified_entity_id:
            if c1.verified_entity_id != c2.verified_entity_id:
                return False

        # CONFLICT GUARD: Don't merge if both have verified_name that disagrees
        # This prevents collapsing two different verified people
        if c1.has_verified() and c2.has_verified():
            # Get verified names from both clusters
            # Fix 20.2: Apply phonetic normalization for Arabic-Hebrew homophones
            # This ensures קאסם vs כאסם (same person) aren't treated as conflict
            v1_names = {normalize_arabic_phonetic(self.normalizer.normalize(m.verified_name))
                        for m in c1.mentions if m.verified_name}
            v2_names = {normalize_arabic_phonetic(self.normalizer.normalize(m.verified_name))
                        for m in c2.mentions if m.verified_name}

            if v1_names and v2_names:
                # Check if any pair has high similarity
                best_v_ratio = 0
                for v1 in v1_names:
                    for v2 in v2_names:
                        if v1 and v2:
                            best_v_ratio = max(best_v_ratio, _token_sort_ratio(v1, v2))

                # If verified names don't match well, don't merge
                if best_v_ratio < 85:
                    return False

        # High similarity on canonicals (using phonetic normalization from Fix 20.4)
        if _token_sort_ratio(name1_phonetic, name2_phonetic) >= cfg.CLUSTER_MERGE_SORT_THRESHOLD:
            return True

        # First + last fuzzy match
        tokens1 = [t for t in self.normalizer.tokenize(name1) if t not in self.noise_tokens]
        tokens2 = [t for t in self.normalizer.tokenize(name2) if t not in self.noise_tokens]

        if len(tokens1) >= 2 and len(tokens2) >= 2:
            # Fix 22.2: Apply phonetic normalization for first/last token comparison
            # Without this, קאסם vs כאסם would get ~75% instead of 100%
            first1_phonetic = normalize_arabic_phonetic(tokens1[0])
            first2_phonetic = normalize_arabic_phonetic(tokens2[0])
            last1_phonetic = normalize_arabic_phonetic(tokens1[-1])
            last2_phonetic = normalize_arabic_phonetic(tokens2[-1])
            first_match = _char_ratio(first1_phonetic, first2_phonetic) >= cfg.CLUSTER_MERGE_FIRST_LAST_THRESHOLD
            last_match = _char_ratio(last1_phonetic, last2_phonetic) >= cfg.CLUSTER_MERGE_FIRST_LAST_THRESHOLD
            if first_match and last_match:
                return True

        # Containment check: all tokens of shorter name appear in longer
        tokens1_set = set(tokens1)
        tokens2_set = set(tokens2)
        if tokens1_set and tokens2_set:
            shorter_set, longer_set = (tokens1_set, tokens2_set) if len(tokens1_set) <= len(tokens2_set) else (tokens2_set, tokens1_set)

            # Safety gates:
            # 1. Shorter must have ≥2 tokens (prevents single-token over-matching)
            # 2. Shorter must be proper subset of longer
            # 3. At most 3 extra tokens in longer
            # 4. token_set_ratio must be very high (when RapidFuzz available)
            # Fix 28.7 (Finding B): Use config threshold instead of hardcoded 95
            if (len(shorter_set) >= 2 and
                shorter_set.issubset(longer_set) and
                len(longer_set) - len(shorter_set) <= 3):
                # Additional check: use token_set_ratio for confirmation
                set_ratio = _token_set_ratio(name1, name2)
                if set_ratio >= cfg.CLUSTER_MERGE_CONTAINMENT_SET_THRESHOLD:
                    return True

        # Debug: log merge rejections
        if self.config.DEBUG_CROSS_PHONE:
            print(f"[MERGE_REJECTED] '{name1}' vs '{name2}'")
            print(f"  token_sort={_token_sort_ratio(name1, name2)}, token_set={_token_set_ratio(name1, name2)}")
            print(f"  tokens1={tokens1}, tokens2={tokens2}")

        return False


    def fuse_by_global_entity_id(
        self,
        clusters: List[EntityCluster]
    ) -> Dict[str, EntityCluster]:
        """Fuse phone-clusters by global_entity_id into entity-level representatives.

        This is THE structural fix: makes entity (not phone-cluster) the
        primary unit for output generation.

        Key: Reuses existing resolver primitives, doesn't create new cascade.
        Returns: Mapping of global_entity_id → NEW fused entity with unified canonical.

        CRITICAL: Always returns NEW objects, never mutates originals.
        """
        from collections import defaultdict

        # Group clusters by global_entity_id (stable iteration order)
        by_entity: Dict[str, List[EntityCluster]] = defaultdict(list)
        for c in sorted(clusters, key=lambda x: x.cluster_id):  # deterministic
            entity_id = c.global_entity_id or c.cluster_id
            by_entity[entity_id].append(c)

        # Fuse each entity group into a single entity-level cluster
        fused: Dict[str, EntityCluster] = {}
        # Iterate in sorted entity_id order for full determinism
        for entity_id in sorted(by_entity.keys()):
            member_clusters = by_entity[entity_id]
            fused[entity_id] = self._fuse_cluster_group(entity_id, member_clusters)

        return fused

    def _fuse_cluster_group(
        self,
        entity_id: str,
        clusters: List[EntityCluster]
    ) -> EntityCluster:
        """Fuse multiple phone-clusters into one entity-level cluster.

        CRITICAL:
        - Reuses existing _check_verified(), _select_canonical() - no new cascade
        - ALWAYS returns a NEW EntityCluster object - never mutates originals
        """
        from copy import copy

        provenance = self._build_provenance(clusters)

        # ALWAYS create new object, even for single cluster (immutability)
        if len(clusters) == 1:
            c = clusters[0]
            return EntityCluster(
                cluster_id=c.cluster_id,  # Keep original cluster_id for single
                phone=c.phone,
                mentions=list(c.mentions),  # shallow copy
                mention_ids=set(c.mention_ids),  # copy
                canonical_name=c.canonical_name,
                display_name=c.display_name,
                resolution_type=c.resolution_type,
                confidence=c.confidence,
                nicknames=set(c.nicknames) if c.nicknames else set(),  # copy
                best_score=c.best_score,
                score_margin=c.score_margin,
                match_evidence=c.match_evidence,
                flags=list(c.flags) if c.flags else [],  # copy
                cross_phone_links=provenance,
                global_entity_id=entity_id,
                verified_entity_id=c.verified_entity_id,  # copy verified_entity_id
                verified_status=c.verified_status,
                verified_id_number=c.verified_id_number,
            )

        # Helper to get verified_entity_id with conflict detection
        def _get_fused_entity_id(clusters_list: List[EntityCluster]) -> Tuple[Optional[str], bool]:
            """Returns (entity_id, has_conflict)."""
            ids = set(c.verified_entity_id for c in clusters_list if c.verified_entity_id is not None)
            if len(ids) == 0:
                return None, False
            elif len(ids) == 1:
                return ids.pop(), False
            else:
                return None, True  # Conflict

        # 1. Check for verified conflict BEFORE fusing
        if self._detect_verified_conflict(clusters):
            # Conflict: pick conservative representative but return NEW object
            rep = self._select_representative(clusters)
            fused_eid, eid_conflict = _get_fused_entity_id(clusters)
            conflict_flags = ['ENTITY_VERIFIED_CONFLICT']
            if eid_conflict:
                conflict_flags.append('ENTITY_ID_CONFLICT')
            return EntityCluster(
                cluster_id=rep.cluster_id,
                phone=rep.phone,
                mentions=list(rep.mentions),
                mention_ids=set(rep.mention_ids),
                canonical_name=rep.canonical_name,
                display_name=rep.display_name,
                resolution_type=rep.resolution_type,
                confidence=rep.confidence,
                nicknames=set(rep.nicknames) if rep.nicknames else set(),
                best_score=rep.best_score,
                score_margin=rep.score_margin,
                match_evidence=rep.match_evidence,
                flags=(list(rep.flags) if rep.flags else []) + conflict_flags,
                cross_phone_links=provenance,
                global_entity_id=entity_id,
                verified_entity_id=fused_eid,
                verified_status=rep.verified_status if fused_eid is not None else None,
                verified_id_number=rep.verified_id_number if fused_eid is not None else None,
            )

        # 2. Check for phonebook conflict (new guardrail)
        if self._detect_phonebook_conflict(clusters):
            rep = self._select_representative(clusters)
            fused_eid, eid_conflict = _get_fused_entity_id(clusters)
            conflict_flags = ['ENTITY_PHONEBOOK_CONFLICT']
            if eid_conflict:
                conflict_flags.append('ENTITY_ID_CONFLICT')
            return EntityCluster(
                cluster_id=rep.cluster_id,
                phone=rep.phone,
                mentions=list(rep.mentions),
                mention_ids=set(rep.mention_ids),
                canonical_name=rep.canonical_name,
                display_name=rep.display_name,
                resolution_type=rep.resolution_type,
                confidence=rep.confidence,
                nicknames=set(rep.nicknames) if rep.nicknames else set(),
                best_score=rep.best_score,
                score_margin=rep.score_margin,
                match_evidence=rep.match_evidence,
                flags=(list(rep.flags) if rep.flags else []) + conflict_flags,
                cross_phone_links=provenance,
                global_entity_id=entity_id,
                verified_entity_id=fused_eid,
                verified_status=rep.verified_status if fused_eid is not None else None,
                verified_id_number=rep.verified_id_number if fused_eid is not None else None,
            )

        # 3. Union all mentions (READ-ONLY - don't mutate originals)
        all_mentions = []
        for c in clusters:
            all_mentions.extend(c.mentions)

        # 4. Normalize verified names BEFORE calling _check_verified
        #    This prevents false VERIFIED_CONFLICT from spelling drift
        normalized_mentions = self._normalize_verified_in_mentions(all_mentions)

        # 5. Resolve using EXISTING primitives (reuse cascade EXACTLY)
        verified_result = self._check_verified(normalized_mentions)
        verified_nicknames: Set[str] = set()  # For merging verified nicknames
        extra_flags: List[str] = []  # For flags from resolution (like 'verified_conflict')
        if verified_result:
            # Create temp cluster and use _resolve_as_verified to stay consistent
            temp_cluster = EntityCluster(
                cluster_id=entity_id,
                phone=clusters[0].phone,
                mentions=normalized_mentions,
                mention_ids={m.mention_id for m in normalized_mentions},
            )
            temp_cluster = self._resolve_as_verified(temp_cluster, verified_result)
            canonical = temp_cluster.canonical_name
            display_name = temp_cluster.display_name
            resolution_type = temp_cluster.resolution_type
            confidence = temp_cluster.confidence
            # FIXED: Use verified scores and evidence from _resolve_as_verified
            best_score = temp_cluster.best_score
            score_margin = temp_cluster.score_margin
            match_evidence = temp_cluster.match_evidence
            # Also capture verified nicknames for merging
            verified_nicknames = temp_cluster.nicknames or set()
            # Capture flags like 'verified_conflict' from _resolve_as_verified
            extra_flags = list(temp_cluster.flags) if temp_cluster.flags else []
        else:
            # Try phonebook - pick best from member clusters
            phonebook_clusters = [c for c in clusters if c.resolution_type == 'PHONEBOOK']
            if phonebook_clusters:
                # Use the exact phonebook result from best member
                best_pb = max(phonebook_clusters, key=lambda c: c.best_score or 0)
                canonical = best_pb.canonical_name
                display_name = best_pb.display_name
                resolution_type = best_pb.resolution_type  # KEEP exact type
                confidence = best_pb.confidence  # KEEP exact confidence
                # FIXED: Use phonebook scores and evidence
                best_score = best_pb.best_score
                score_margin = best_pb.score_margin
                match_evidence = best_pb.match_evidence
            else:
                # REUSE: Call _select_canonical on all mentions
                canonical = self._select_canonical(all_mentions)
                # Find raw version of the selected canonical mention
                display_name = canonical  # fallback
                for m in all_mentions:
                    if m.normalized == canonical:
                        display_name = m.raw_text
                        break
                resolution_type = 'INFERRED'
                confidence = 'LOW'
                # For inferred, use best scores from strongest member
                best_cluster = max(clusters, key=lambda c: c.best_score or 0)
                best_score = best_cluster.best_score
                score_margin = best_cluster.score_margin
                match_evidence = f"fused:{len(clusters)}_clusters"

        # 6. Build fused entity (NEW object)
        primary_phone = clusters[0].phone

        # Merge all nicknames from all clusters + verified nicknames
        merged_nicknames: Set[str] = set()
        for c in clusters:
            if c.nicknames:
                merged_nicknames.update(c.nicknames)
        # Also include verified nicknames from _resolve_as_verified
        merged_nicknames.update(verified_nicknames)

        # Combine base flag with any extra flags from resolution
        fused_flags = ['ENTITY_FUSED'] + extra_flags

        # Collect all verified_entity_ids from member clusters
        # Handle conflict: if multiple different IDs, don't propagate any (set to None)
        entity_ids_set = set(
            c.verified_entity_id for c in clusters
            if c.verified_entity_id is not None
        )

        if len(entity_ids_set) == 0:
            fused_verified_entity_id = None
        elif len(entity_ids_set) == 1:
            fused_verified_entity_id = entity_ids_set.pop()
        else:
            # CONFLICT: Multiple different entity IDs in same fused entity
            # Don't propagate wrong ID - set to None and flag
            fused_verified_entity_id = None
            fused_flags.append('ENTITY_ID_CONFLICT')
            if self.config.DEBUG_CROSS_PHONE:
                print(f"  [ENTITY_ID_CONFLICT] Fused entity {entity_id} has conflicting IDs: {entity_ids_set}")

        # Derive fused status/id_number from the cluster that has the entity_id
        fused_verified_status = None
        fused_verified_id_number = None
        if fused_verified_entity_id:
            for c in clusters:
                if c.verified_entity_id == fused_verified_entity_id:
                    fused_verified_status = c.verified_status
                    fused_verified_id_number = c.verified_id_number
                    break

        # Fix 11.2: Use normalized_mentions instead of all_mentions
        # Previously used all_mentions (dirty) but canonical_name was derived from
        # normalized_mentions (clean), causing internal inconsistency where
        # canonical_name didn't match the actual mentions in the cluster.
        fused = EntityCluster(
            cluster_id=entity_id,  # Use entity_id as cluster_id for fused
            phone=primary_phone,
            mentions=normalized_mentions,
            mention_ids={m.mention_id for m in normalized_mentions},
            canonical_name=canonical,
            display_name=display_name,
            resolution_type=resolution_type,
            confidence=confidence,
            nicknames=merged_nicknames,
            flags=fused_flags,
            cross_phone_links=provenance,
            global_entity_id=entity_id,
            verified_entity_id=fused_verified_entity_id,
            verified_status=fused_verified_status,
            verified_id_number=fused_verified_id_number,
        )

        # Use the scores and evidence determined above
        fused.best_score = best_score
        fused.score_margin = score_margin
        fused.match_evidence = match_evidence

        return fused

    def _detect_verified_conflict(self, clusters: List[EntityCluster]) -> bool:
        """Detect if clusters have conflicting verified names.

        Returns True if fusion should be blocked/conservative.
        """
        verified_names = set()
        for c in clusters:
            for m in c.mentions:
                if m.verified_name:
                    verified_names.add(self.normalizer.normalize(m.verified_name))

        if len(verified_names) <= 1:
            return False

        # Check pairwise similarity - if any pair is very different, conflict
        names_list = sorted(verified_names)
        for i, n1 in enumerate(names_list):
            for n2 in names_list[i+1:]:
                # Fix 10.2: Apply phonetic normalization before comparison.
                # Arabic transliterations like קאסם vs כאסם (Qassem) are the SAME person
                # but only 75% similar due to ק/כ substitution. Normalizing homophones
                # before comparison prevents false VERIFIED_CONFLICT flags.
                n1_phonetic = normalize_arabic_phonetic(n1)
                n2_phonetic = normalize_arabic_phonetic(n2)
                # Threshold aligned with normalization (both use 85%) to prevent
                # "Gap of Tolerance" bug where names merge but don't normalize.
                sim = _token_sort_ratio(n1_phonetic, n2_phonetic)
                if sim < 85:  # Very different verified names
                    return True

        return False

    def _detect_phonebook_conflict(self, clusters: List[EntityCluster]) -> bool:
        """Detect if clusters have conflicting high-confidence phonebook matches.

        Returns True if ≥2 clusters have PHONEBOOK resolution with high scores
        but very different canonicals.
        """
        phonebook_clusters = [
            c for c in clusters
            if c.resolution_type == 'PHONEBOOK' and (c.best_score or 0) >= 0.80
        ]

        if len(phonebook_clusters) < 2:
            return False

        # Check pairwise similarity of canonicals
        for i, c1 in enumerate(phonebook_clusters):
            for c2 in phonebook_clusters[i+1:]:
                n1 = self.normalizer.normalize(c1.canonical_name)
                n2 = self.normalizer.normalize(c2.canonical_name)
                if n1 and n2:
                    # Fix 20.1: Apply phonetic normalization for Arabic-Hebrew homophones
                    # This ensures קאסם vs כאסם (same person) aren't flagged as conflict
                    n1_phonetic = normalize_arabic_phonetic(n1)
                    n2_phonetic = normalize_arabic_phonetic(n2)
                    sim = _token_sort_ratio(n1_phonetic, n2_phonetic)
                    if sim < 80:  # Very different phonebook matches
                        return True

        return False

    def _normalize_verified_in_mentions(
        self,
        mentions: List[NameMention]
    ) -> List[NameMention]:
        """Create wrapper mentions with normalized verified_name.

        This prevents false VERIFIED_CONFLICT from spelling drift like
        'אלמשהראוי' vs 'אלמשהראווי'.

        FIXED: Uses fuzzy similarity (union-find) to group verified names,
        not exact normalized equality.

        Returns list of shallow-copy mentions with normalized verified_name.
        Original mentions are NOT mutated.
        """
        from copy import copy

        # Collect all unique verified names
        verified_names: List[str] = []
        for m in mentions:
            if m.verified_name and m.verified_name not in verified_names:
                verified_names.append(m.verified_name)

        if len(verified_names) <= 1:
            # No normalization needed - 0 or 1 unique verified name
            return mentions

        # Union-Find to group similar verified names
        parent: Dict[str, str] = {n: n for n in verified_names}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])  # path compression
            return parent[x]

        # Count frequency of each verified name for better canonical selection
        name_freq: Dict[str, int] = {}
        for m in mentions:
            if m.verified_name:
                name_freq[m.verified_name] = name_freq.get(m.verified_name, 0) + 1

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # FIXED: Prefer most frequent name as root, then shorter
                # (frequency is better than length - kunya/truncated forms may be shorter)
                freq_a, freq_b = name_freq.get(ra, 0), name_freq.get(rb, 0)
                if freq_b > freq_a or (freq_b == freq_a and len(rb) < len(ra)):
                    ra, rb = rb, ra
                parent[rb] = ra

        # Group by fuzzy similarity (token_sort_ratio >= 85)
        # IMPORTANT: Threshold aligned with _detect_verified_conflict (also 85%)
        # to prevent "Gap of Tolerance" bug where names merge but don't normalize,
        # causing false VERIFIED_CONFLICT flags for names with 85-89% similarity.
        # Fix 10.2: Apply phonetic normalization before comparison for Arabic homophones.
        for i, n1 in enumerate(verified_names):
            norm1 = normalize_arabic_phonetic(self.normalizer.normalize(n1))
            for n2 in verified_names[i+1:]:
                norm2 = normalize_arabic_phonetic(self.normalizer.normalize(n2))
                if _token_sort_ratio(norm1, norm2) >= 85:
                    union(n1, n2)

        # Build mapping from each verified name to its group's canonical
        # (the root of its union-find component)
        name_to_canonical: Dict[str, str] = {}
        for n in verified_names:
            root = find(n)
            name_to_canonical[n] = root

        # Create copies with canonical verified_name
        result = []
        for m in mentions:
            if m.verified_name:
                canonical_form = name_to_canonical.get(m.verified_name, m.verified_name)
                if canonical_form != m.verified_name:
                    # Create shallow copy with replaced verified_name
                    m_copy = copy(m)
                    m_copy.verified_name = canonical_form
                    result.append(m_copy)
                else:
                    result.append(m)
            else:
                result.append(m)

        return result

    def _build_provenance(self, clusters: List[EntityCluster]) -> List[Dict[str, str]]:
        """Build provenance list for cross_phone_links JSON field."""
        return [
            {
                'phone_cluster_id': c.cluster_id,
                'phone': c.phone,
                'canonical': c.canonical_name,
                'type': c.resolution_type,
            }
            for c in sorted(clusters, key=lambda x: x.cluster_id)
        ]

    def _select_representative(
        self,
        clusters: List[EntityCluster]
    ) -> EntityCluster:
        """Pick best cluster to represent the entity.

        Priority: CALL_VERIFIED > PHONEBOOK > largest INFERRED
        """
        if len(clusters) == 1:
            return clusters[0]

        # Priority 1: CALL_VERIFIED with most mentions
        verified = [c for c in clusters if c.resolution_type == "CALL_VERIFIED"]
        if verified:
            return max(verified, key=lambda c: len(c.mentions))

        # Priority 2: PHONEBOOK
        phonebook = [c for c in clusters if c.resolution_type == "PHONEBOOK"]
        if phonebook:
            # Prefer the one with highest score
            return max(phonebook, key=lambda c: c.best_score or 0)

        # Priority 3: Largest INFERRED cluster
        return max(clusters, key=lambda c: len(c.mentions))


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

OUTPUT_COLUMNS = [
    # --- Diagnostic (first 9 — screenshot-friendly) ---
    'speaker_phone', 'raw_clean_name',
    'verified_name_call', 'verified_nicknames_call',
    'resolved_name', 'resolved_type',
    'global_entity_id', 'best_score', 'score_margin',
    # --- Call context ---
    'call_id', 'date', 'side', 'other_phone',
    # --- Segment ---
    'segment_index', 'segment_count', 'is_blmz',
    # --- Entity linking ---
    'verified_entity_id',
    'verified_status', 'verified_id_number',
    # --- Clustering ---
    'cluster_id', 'cluster_size',
    # --- Scoring ---
    'confidence',
    # --- Evidence ---
    'match_evidence', 'cross_phone_links', 'flags',
]


class OutputGenerator:
    """Generate output DataFrame from resolved clusters."""

    def __init__(self, config: Config):
        self.config = config

    def generate(
        self,
        clusters: List[EntityCluster],
        mentions: List[NameMention],
        entity_representatives: Dict[str, EntityCluster] = None
    ) -> pd.DataFrame:
        """Generate output DataFrame from clusters and mentions.

        Args:
            clusters: List of all clusters
            mentions: List of all mentions
            entity_representatives: Optional mapping of global_entity_id →
                representative cluster for entity-level canonical names.
                If provided, resolved_name/resolved_type/confidence come from
                the representative, ensuring consistency across phones.
        """
        cluster_by_mention: Dict[str, EntityCluster] = {}
        for cluster in clusters:
            for mention_id in cluster.mention_ids:
                cluster_by_mention[mention_id] = cluster

        rows = []
        for mention in mentions:
            cluster = cluster_by_mention.get(mention.mention_id)
            row = self._generate_row(mention, cluster, entity_representatives)
            rows.append(row)

        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        
        # DIAGNOSTIC: Check verified_entity_id in output
        eid_col_idx = OUTPUT_COLUMNS.index('verified_entity_id')
        non_empty_eids = sum(1 for r in rows if r[eid_col_idx] and str(r[eid_col_idx]).strip())
        print(f"[DIAG OUTPUT] verified_entity_id non-empty: {non_empty_eids}/{len(rows)}")
        
        # Also check clusters
        clusters_with_eid = sum(1 for c in clusters if c.verified_entity_id)
        print(f"[DIAG OUTPUT] Clusters with verified_entity_id: {clusters_with_eid}/{len(clusters)}")
        
        if entity_representatives:
            reps_with_eid = sum(1 for e in entity_representatives.values() if e.verified_entity_id)
            print(f"[DIAG OUTPUT] Entity representatives with verified_entity_id: {reps_with_eid}/{len(entity_representatives)}")
        
        return df

    def _generate_row(
        self,
        mention: NameMention,
        cluster: EntityCluster = None,
        entity_representatives: Dict[str, EntityCluster] = None
    ) -> List[Any]:
        """Generate a single output row for a mention.

        KEY CHANGE: ALL entity-level fields (resolved_name, resolved_type,
        confidence, scores, cross_phone_links, flags) come from entity_representatives,
        not from phone-cluster. This ensures internal row consistency.
        """
        if cluster is None:
            cluster_id = ''
            global_entity_id = ''
            cluster_size = 1
            resolved_name = mention.raw_text
            resolved_type = 'UNRESOLVED'
            confidence = 'LOW'
            best_score = 0.0
            score_margin = 0.0
            match_evidence = ''
            cross_phone_links = '[]'  # Empty JSON array when no cluster
            entity_flags = ['unresolved']
            # Fallback: use mention-level verified_entity_id if available
            verified_entity_id = mention.verified_entity_id or ''
            verified_status = mention.verified_status or ''
            verified_id_number = mention.verified_id_number or ''
        else:
            cluster_id = cluster.cluster_id
            global_entity_id = cluster.global_entity_id
            cluster_size = cluster.size

            # Entity Fusion: use ALL entity-level fields from fused entity
            # This ensures consistency: resolved_name + scores come from same source
            if entity_representatives and global_entity_id and global_entity_id in entity_representatives:
                entity = entity_representatives[global_entity_id]
                entity_flags = list(entity.flags) if entity.flags else []

                # CRITICAL FIX: If entity has conflict flags, use phone-cluster fields
                # This makes the guardrail protective, not just diagnostic
                has_conflict = any(f in entity_flags for f in [
                    'ENTITY_VERIFIED_CONFLICT',
                    'ENTITY_PHONEBOOK_CONFLICT'
                ])

                if has_conflict:
                    # Conflict detected - keep per-phone canonicals, but still show provenance
                    resolved_name = cluster.display_name or cluster.canonical_name
                    resolved_type = cluster.resolution_type
                    confidence = cluster.confidence
                    best_score = cluster.best_score
                    score_margin = cluster.score_margin
                    match_evidence = cluster.match_evidence
                    # Still write cross_phone_links for debugging
                    cross_phone_links = json.dumps(entity.cross_phone_links, ensure_ascii=False) if entity.cross_phone_links else '[]'
                    # Merge cluster flags with entity conflict flag (for visibility)
                    entity_flags = (list(cluster.flags) if cluster.flags else []) + [
                        f for f in entity_flags if 'CONFLICT' in f
                    ]
                    # Use entity verified_entity_id (propagated through fusion)
                    verified_entity_id = entity.verified_entity_id or cluster.verified_entity_id or mention.verified_entity_id or ''
                    verified_status = entity.verified_status or cluster.verified_status or mention.verified_status or ''
                    verified_id_number = entity.verified_id_number or cluster.verified_id_number or mention.verified_id_number or ''
                else:
                    # Normal fusion - use entity-level fields
                    resolved_name = entity.display_name or entity.canonical_name
                    resolved_type = entity.resolution_type
                    confidence = entity.confidence
                    best_score = entity.best_score
                    score_margin = entity.score_margin
                    match_evidence = entity.match_evidence
                    cross_phone_links = json.dumps(entity.cross_phone_links, ensure_ascii=False) if entity.cross_phone_links else '[]'
                    # Use entity verified_entity_id (propagated through fusion)
                    verified_entity_id = entity.verified_entity_id or mention.verified_entity_id or ''
                    verified_status = entity.verified_status or mention.verified_status or ''
                    verified_id_number = entity.verified_id_number or mention.verified_id_number or ''
            else:
                # Fallback to cluster-level (single-cluster entity or missing)
                resolved_name = cluster.display_name or cluster.canonical_name
                resolved_type = cluster.resolution_type
                confidence = cluster.confidence
                best_score = cluster.best_score
                score_margin = cluster.score_margin
                match_evidence = cluster.match_evidence
                cross_phone_links = json.dumps(cluster.cross_phone_links, ensure_ascii=False) if cluster.cross_phone_links else '[]'
                entity_flags = list(cluster.flags) if cluster.flags else []
                # Use cluster verified_entity_id with mention fallback
                verified_entity_id = cluster.verified_entity_id or mention.verified_entity_id or ''
                verified_status = cluster.verified_status or mention.verified_status or ''
                verified_id_number = cluster.verified_id_number or mention.verified_id_number or ''

        mention_flags = []
        if mention.segment_count > 1:
            mention_flags.append('multi_speaker')
        if mention.verified_name:
            mention_flags.append('verified_assigned')
        if mention.is_blmz:
            mention_flags.append('blmz')

        # Use entity_flags instead of cluster.flags for consistency
        mention_flags.extend(entity_flags)

        # Propagate verified fields to ALL rows that belong to a verified entity/cluster.
        # Otherwise, only the single "verified" mention keeps these columns populated, which is misleading.
        propagated_verified_name = mention.verified_name or ''
        propagated_verified_nicks = mention.verified_nicknames or ''

        if cluster is not None and (not propagated_verified_name):
            # Prefer fused entity representative if available
            if entity_representatives and global_entity_id and global_entity_id in entity_representatives:
                e = entity_representatives[global_entity_id]
                if e.resolution_type in ('CALL_VERIFIED', 'VERIFIED_CONFLICT', 'PHONEBOOK'):
                    propagated_verified_name = e.display_name or e.canonical_name or ''
                    if (not propagated_verified_nicks) and e.nicknames:
                        propagated_verified_nicks = ', '.join(sorted(e.nicknames))
            # Fallback to per-phone cluster
            if (not propagated_verified_name) and cluster.resolution_type in ('CALL_VERIFIED', 'VERIFIED_CONFLICT', 'PHONEBOOK'):
                propagated_verified_name = cluster.display_name or cluster.canonical_name or ''
                if (not propagated_verified_nicks) and cluster.nicknames:
                    propagated_verified_nicks = ', '.join(sorted(cluster.nicknames))

        row = [
            # --- Diagnostic (first 9 — screenshot-friendly) ---
            mention.phone,
            mention.original_field,
            propagated_verified_name,
            propagated_verified_nicks,
            resolved_name,
            resolved_type,
            global_entity_id,
            round(best_score or 0.0, 4),
            round(score_margin or 0.0, 4),
            # --- Call context ---
            mention.call_id,
            mention.date or '',
            mention.side,
            mention.other_phone,
            # --- Segment ---
            mention.segment_index,
            mention.segment_count,
            mention.is_blmz,
            # --- Entity linking ---
            verified_entity_id,
            verified_status or '',
            verified_id_number or '',
            # --- Clustering ---
            cluster_id,
            cluster_size,
            # --- Scoring ---
            confidence,
            # --- Evidence ---
            match_evidence or '',
            cross_phone_links,
            json.dumps(list(set(mention_flags)), ensure_ascii=False),
        ]

        return row


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class EntityResolutionPipeline:
    """Main pipeline for Hebrew entity resolution."""

    def __init__(self, config: Config = None, cube2_df: pd.DataFrame = None):
        self.config = config or Config()
        self.config.validate()

        self.exploder = MentionExploder(self.config)
        self.vectorizer = EntityVectorizer(self.config)
        self.cube2_matcher = Cube2Matcher(self.config, cube2_df)
        self.resolver = EntityResolver(self.config, self.cube2_matcher)
        self.output_generator = OutputGenerator(self.config)

        self.scorer: Optional[SimilarityScorer] = None
        self.graph_builder: Optional[SimilarityGraph] = None
        self.clusterer = HacClusterer(self.config)

        self.mentions: List[NameMention] = []
        self.clusters: List[EntityCluster] = []

        self.api_client = None
        if self.config.API_ENABLED and HAS_REQUESTS and self.config.API_A_URL:
            self.api_client = YanisAPIClient(
                self.config.API_A_URL, self.config.API_A_TOKEN,
                self.config.API_TIMEOUT, self.config.API_BATCH_MAX_NAMES)
        self.api_decisions = []

    def run(self, cube1_df: pd.DataFrame) -> pd.DataFrame:
        """Run the full entity resolution pipeline."""
        # Stage 0: Schema adapter
        cube1_df = self._adapt_schema(cube1_df)

        # Stage 1: Mention explosion
        self.mentions = self.exploder.explode_dataframe(cube1_df)

        if not self.mentions:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Stage 2-3: Vectorization
        self.vectorizer.fit_transform(self.mentions)
        # Enrich cube2 contacts with corpus stats once vectorizer is fitted (v9)
        self.cube2_matcher.finalize_contact_quality(self.vectorizer)

        # Build gates and indices
        ambiguity_gate = AmbiguityGate(self.mentions)
        # Fix 13.2: Pass ambiguity_gate to resolver for Stage 6 phonebook checks
        self.resolver.ambiguity_gate = ambiguity_gate
        # Pass normalizer to ensure nickname normalization matches mention normalization
        phone_alias_index = PhoneAliasIndex(self.mentions, self.exploder.normalizer)

        # Initialize scorer and graph builder
        self.scorer = SimilarityScorer(
            self.config, self.vectorizer, ambiguity_gate, phone_alias_index
        )
        self.graph_builder = SimilarityGraph(
            self.config, self.vectorizer, self.scorer
        )

        # Stage 4: Build similarity graphs per phone
        phone_graphs = self.graph_builder.build_phone_graphs(self.mentions)

        # Stage 5: Clustering per phone
        all_labels: Dict[str, str] = {}

        for phone in sorted(phone_graphs.keys()):
            graph = phone_graphs[phone]
            phone_mentions = [
                m for m in self.mentions
                if m.phone == phone and m.mention_id in graph.nodes()
            ]
            labels = self.clusterer.cluster_with_constraints(graph, phone_mentions)

            for mention_id, cluster_id in labels.items():
                all_labels[mention_id] = f"{phone}__{cluster_id}"

        # Handle isolated mentions
        for m in self.mentions:
            if m.mention_id not in all_labels:
                all_labels[m.mention_id] = f"{m.phone}__isolated_{m.mention_id}"

        # Stage 6: Resolution
        self.clusters = self._resolve_clusters(all_labels)

        # Stage 6.2: Yanis verified-anchor API attachment (Fix 30.1)
        # Injects entity_id onto unresolved clusters so Stage 7 Phase 0 can hard-link
        # them, bypassing the cohesion gate that blocks mention-level mismatches.
        # Must run BEFORE 6.25 so that newly-attached entity_ids are available
        # for cube2 bridging decisions.
        if self.api_client and self.config.API_ANCHOR_ATTACH_ENABLED:
            self.clusters = self.resolver.merge_clusters_by_verified_api_anchors(
                self.clusters, api_client=self.api_client, ambiguity_gate=ambiguity_gate)

        # Stage 6.25: cube2-driven intra-phone bridging (v9)
        # Bridges fragmented clusters like {first} + {last} when both map to the same high-quality phonebook contact.
        self.clusters = self.resolver.merge_clusters_by_cube2_anchors(
            self.clusters, ambiguity_gate=ambiguity_gate
        )

        # Stage 6.5: Cluster-level merge pass (Fix 3.4)
        # Merge clusters with very similar canonicals that didn't connect at mention level
        self.clusters = self.resolver.merge_similar_clusters(self.clusters)

        # Stage 6.75: Controlled kunya alias propagation (same phone, anchor-backed)
        # Recover unresolved kunya-only clusters when exactly one strong local
        # entity anchor shares the kunya alias set.
        self.clusters = self.resolver.propagate_kunya_alias_on_phone(self.clusters)

        # Stage 6.8: Noise detection (structural, no API)
        if self.config.NOISE_DETECTION_ENABLED:
            self.clusters = self.resolver._detect_noise_clusters(self.clusters)

        # Stage 7: Global Cross-Phone Linking (v8 Architecture)
        # Uses ClusterDSU with constraint enforcement to merge clusters across phones
        # Replaces old find_cross_phone_links with constraint-aware DSU merge
        # v8 Fix C: Pass scorer for consistent cohesion gate scoring
        # Fix 26.9: Pass ambiguity_gate for global singleton safety check
        cluster_dsu = self.resolver.global_cluster_merge(
            self.clusters, scorer=self.scorer, ambiguity_gate=ambiguity_gate,
            api_client=self.api_client)
        self._apply_dsu_to_clusters(cluster_dsu, self.clusters)

        # Debug: Show entity grouping to verify Stage 7 quality
        if self.config.DEBUG_CROSS_PHONE:
            self._debug_entity_grouping(self.clusters)

        # Stage 8: Entity Fusion - THE KEY STRUCTURAL FIX
        # Creates mapping of global_entity_id → representative cluster
        # This ensures consistent resolved_name across all phones for same entity
        entity_representatives = self.resolver.fuse_by_global_entity_id(self.clusters)

        # Generate output with entity-level canonical names
        output_df = self.output_generator.generate(
            self.clusters, self.mentions, entity_representatives
        )

        if self.api_client:
            self.api_decisions = self.api_client.decisions
            if self.config.DEBUG_CROSS_PHONE:
                print(self.api_client.log_summary())

        return output_df

    def _debug_entity_grouping(self, clusters: List[EntityCluster]) -> None:
        """Print entity grouping for debugging Stage 7 quality.

        Shows which phone-clusters are linked to each global_entity_id,
        helping verify that cross-phone linking is working correctly.
        """
        from collections import defaultdict

        by_entity: Dict[str, List[str]] = defaultdict(list)
        for c in clusters:
            entity_id = c.global_entity_id or c.cluster_id
            by_entity[entity_id].append(f"{c.phone}:{c.canonical_name}")

        # Find entities that span multiple phones
        multi_phone = {
            k: v for k, v in by_entity.items()
            if len(set(x.split(':')[0] for x in v)) > 1
        }

        print(f"DEBUG ENTITY_FUSION: {len(multi_phone)} entities span multiple phones")
        for entity_id, members in list(multi_phone.items())[:10]:  # Sample 10
            phones = set(x.split(':')[0] for x in members)
            print(f"  {entity_id}: {len(phones)} phones, members={members[:3]}{'...' if len(members) > 3 else ''}")

    def _apply_dsu_to_clusters(self, dsu: ClusterDSU, clusters: List[EntityCluster]) -> None:
        """Apply ClusterDSU results to assign global_entity_id to each cluster.

        This method:
        1. Gets all groups from the DSU (each group = one global entity)
        2. Assigns a sequential global_entity_id to each group
        3. Stores merge_reason on each cluster for observability
        """
        # Get groups from DSU: {root: [cluster_ids]}
        groups = dsu.get_groups()

        # Create deterministic global entity IDs based on group membership.
        # IMPORTANT: do NOT base IDs on DSU roots or sequential ordering, because the chosen
        # root can change due to union-by-rank, which would make entity IDs unstable across runs.
        import hashlib

        def _stable_entity_id(members: List[str]) -> str:
            sig = "|".join(sorted(members))
            digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
            return f"E{digest}"

        # Build cluster_id → global_entity_id mapping
        cluster_to_entity = {}
        for root, members in groups.items():
            entity_id = _stable_entity_id(members)
            for cluster_id in members:
                cluster_to_entity[cluster_id] = entity_id

        # Apply to clusters and store merge reason
        for c in clusters:
            if c.cluster_id in cluster_to_entity:
                c.global_entity_id = cluster_to_entity[c.cluster_id]
                # Store merge reason for observability (if available)
                if c.cluster_id in dsu.merge_reasons:
                    c.merge_reason = dsu.merge_reasons[c.cluster_id]
            else:
                # Cluster not in DSU (shouldn't happen, but fallback)
                c.global_entity_id = f"E_isolated_{c.cluster_id}"

        # Store blocked merges info on resolver for observability
        self.resolver.blocked_merges = dsu.blocked_merges

    def _adapt_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt input schema to expected column names.

        Maps point(1).py output columns to entity_resolution expected columns:
        - original_call_id → call_id
        - num_A → pstn_A
        - num_B → pstn_B
        - name_cleaned_A → clean_name_A
        - name_cleaned_B → clean_name_B
        - verified_nickname_A → verified_nicknames_A
        - verified_nickname_B → verified_nicknames_B

        This prevents production failures when upstream column names differ.
        """
        column_aliases = {
            # Call ID mapping
            'original_call_id': 'call_id',
            # Phone number mappings
            'num_A': 'pstn_A',
            'num_B': 'pstn_B',
            # Name mappings
            'name_cleaned_A': 'clean_name_A',
            'name_cleaned_B': 'clean_name_B',
            # Nickname mappings
            'verified_nickname_A': 'verified_nicknames_A',
            'verified_nickname_B': 'verified_nicknames_B',
            # Status mappings
            'status_A': 'verified_status_A',
            'status_B': 'verified_status_B',
            # ID number mappings
            'id_number_A': 'verified_id_number_A',
            'id_number_B': 'verified_id_number_B',
            'tz_A': 'verified_id_number_A',
            'tz_B': 'verified_id_number_B',
        }

        rename_map = {
            old: new for old, new in column_aliases.items()
            if old in df.columns and new not in df.columns
        }

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _resolve_clusters(self, labels: Dict[str, str]) -> List[EntityCluster]:
        """Resolve all clusters using the priority cascade."""
        by_cluster: Dict[str, List[NameMention]] = defaultdict(list)
        mention_by_id = {m.mention_id: m for m in self.mentions}

        for mention_id, cluster_id in labels.items():
            mention = mention_by_id.get(mention_id)
            if mention:
                by_cluster[cluster_id].append(mention)
                mention.cluster_id = cluster_id

        clusters = []
        for cluster_id, mentions in by_cluster.items():
            phone = mentions[0].phone if mentions else ''
            cluster = self.resolver.resolve_cluster(cluster_id, phone, mentions)
            clusters.append(cluster)

        return clusters


# ============================================================================
# FRAMEWORK v7 ENTRY POINT
# ============================================================================

def main(cubes):
    """
    Framework v7 compliant entry point.

    Args:
        cubes: Dictionary of DataFrames keyed by cube name
               - cube1: Call data (required)
               - cube2: Phone contacts (optional)

    Returns:
        DataFrame with resolved entities (25 columns, one row per mention)
    """
    config = Config()

    # Validate required cube
    if config.CUBE_CALLS not in cubes:
        return create_error_df('MissingCube', f'Required cube "{config.CUBE_CALLS}" not found')

    cube1_df = cubes[config.CUBE_CALLS]

    if cube1_df.empty:
        return create_error_df('EmptyData', f'Cube "{config.CUBE_CALLS}" is empty')

    # Get optional cube2
    cube2_df = cubes.get(config.CUBE_CONTACTS)
    if cube2_df is not None and cube2_df.empty:
        cube2_df = None

    try:
        # Run pipeline
        pipeline = EntityResolutionPipeline(config=config, cube2_df=cube2_df)
        result = pipeline.run(cube1_df)

        print(f"Entity Resolution Complete:")
        print(f"  - Input rows: {len(cube1_df)}")
        print(f"  - Output mentions: {len(result)}")
        print(f"  - Clusters: {result['cluster_id'].nunique() if not result.empty else 0}")

        return sanitize_dataframe(result)

    except Exception as e:
        return create_error_df(type(e).__name__, str(e))


# ============================================================================
# EXECUTION
# ============================================================================

if 'cubes' in globals():
    result = main(cubes)



