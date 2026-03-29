# -*- coding: utf-8 -*-
r"""
Hebrew Call Record Name Extractor - WITH VERIFIED ENTITY INTEGRATION
==========================================================================
Version: 2.9.0 - Extended Role Keywords & Removers
Purpose: Extract clean person names from complex Hebrew call transcripts
         + Integrate verified entity names from KIK system

Changes from v2.8.9:
✓ NEW: Role keywords: ראש, לשעבר, סמח״ט, סאג״ד, מח״ט, מפקצ
✓ NEW: All abbreviations now have no-quote (bare) variants (e.g., מגד, סמחט)
✓ NEW: ה-prefix auto-matching (המג״ד, הסמח״ט, etc.) via pattern modification
✓ NEW: Removers added: ראו, לפי, בהשוואת, ע״י (with all quote variants)
✓ NEW: ABBREVIATION_REBUILDS updated with ה-prefix variants
✓ FIX: רחל bare form removed - it's a common female name (false positive)

Changes from v2.8.8:
✓ NEW: Treat ׳׳ (two geresh) exactly like ״ (gershayim)
✓ NEW: Early normalization in parse_content_for_names() BEFORE pattern matching
✓ NEW: Also normalizes '' (two ASCII apostrophes) → ״
✓ FIX: Patterns like ב[״׳"\']ר now correctly match ב׳׳ר after normalization

Changes from v2.8.7:
✓ FIX: Handle ״ used as word separator (מחמד״אלעג׳לה״זוהה״עפ״ק)
✓ FIX: Convert ALL ״ to spaces, then rebuild known abbreviations
✓ FIX: Use regex word boundaries in rebuild to prevent corrupting names
✓ FIX: Removed bare short forms (מג, מפ, etc.) from ROLE_KEYWORDS - they caused false positives
✓ FIX: Added Hebrew geresh (׳) variants to ALL role keywords for complete quote coverage
✓ FIX: All cleaning patterns include ׳ (Hebrew geresh) for עפ׳ק variants

Changes from v2.8.6:
✓ FIX: Quote preservation - no longer destroys gershayim, detection handles all variants
✓ NEW: Robust BLMZ detection - catches בלמ״ז/בלמז/בלמ ז/בלמזית anywhere in text
✓ NEW: Speaker placeholder detection - דובר/דוברת א/ב with any quote style
✓ FIX: junk_tokens pattern now catches עפ ק, עפ,ק (space/comma separators)
✓ FIX: br_all_variants pattern now catches `, - ב״ר` (any space/comma/dash combo)
✓ NEW: Added role keyword: מאייש
✓ NEW: Added metadata phrases: כיום, DTMF, שיח קודם (truncate from these)

Changes from v2.8.5:
✓ FIX: Mission-safe quote normalization (superseded by v2.8.8 approach)
✓ FIX: Added Hebrew geresh ׳ (U+05F3) to all quote character classes

Changes from v2.8.1:
✓ NEW: DEBUG_SWAP_CONSISTENCY flag to enable swap validation
✓ NEW: validate_swap_consistency() function to detect phone/name swap mismatches
✓ NEW: Logs inconsistencies with full details (call_id, before/after values)
✓ NEW: Adds SWAP_INCONSISTENCY_DETECTED to metadata when issue detected
✓ FIX: Regex pattern now allows comma before role keywords (was only space)
✓ NEW: Added missing role keywords: רמ״ד, תצפיות
✓ NEW: Cleaning patterns for percentage (67% עפ״ק) and partial verification (זוהה ע״י)
✓ NEW: Speaker placeholders (דובר א/ב) automatically replaced with בלמ״ז

Changes from v2.8.0:
✓ NEW: Invalidation rule for משמש/משמשת (clears name_role + name_cleaned)
✓ NEW: Added role keywords: רל״ש, רל״שו, סייען, מחליף/מחליפו/מחליפה

Changes from v2.7.7:
✓ REMOVED: content_full, side_A_full_line, side_B_full_line (saves 3 cols)
✓ REMOVED: raw_extraction_A, raw_extraction_B (saves 2 cols)
✓ COMBINED: extraction_notes, swap_reason, br_pattern_found, had_swap → metadata
✓ NEW: cube3 integration (enhancement table - id, title, nicknames)
✓ NEW: cube4 integration (Script 1 output - call_id, phone, original_id)
✓ NEW: verified_name_A, verified_name_B, verified_nickname_A, verified_nickname_B
✓ NEW: verified_entity_id_A, verified_entity_id_B

Column count: 24 (under 25 limit)
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, List, Set, Dict, Any
from dataclasses import dataclass, field

# ============================================================================
# CONFIGURATION - CUSTOMIZE THIS SECTION FOR YOUR DATA
# ============================================================================

class Config:
    """
    System configuration for Point script.
    
    INSTRUCTIONS:
    1. Update INPUT_CUBE_* if your cube names differ
    2. Update COL_* for cube1/cube2 input columns
    3. Update CUBE3_COL_*/CUBE4_COL_* for enhancement/Script1 column names
    """
    
    # ----- Cube Names -----
    INPUT_CUBE = 'cube1'    # Raw call data
    INPUT_CUBE_2 = 'cube5'  # Config with main_number
    INPUT_CUBE_3 = 'cube3'  # Enhancement table (entity data: id, title, nicknames)
    INPUT_CUBE_4 = 'cube4'  # Script 1 output (call_id, phone, original)

    # ----- cube1 Input Columns -----
    COL_CALL_ID = '6bfe4f0c-6176-48e0-a421-01dec3498f2e'
    COL_NUM_A = '3d79b534-559c-460c-88e8-05667cb83ed2'
    COL_NUM_B = '335d9912-987f-4453-8041-d059bfd6193b'
    COL_CONTENT = 'IntelliItemContent'
    COL_DATE = 'IntelliItemCreationTime'
    COL_MAIN_NUMBER = 'SHLUHA'  # From cube5

    # ----- cube3 (Enhancement Table) Columns -----
    # Enhancement table columns:
    # - ~apak_id: join key (links to cube4.original)
    # - id: verified entity ID (output as verified_entity_id_A/B)
    # - title: entity name
    # - nicknames: entity nicknames
    CUBE3_COL_APAK_ID = '~apak_id'  # Join key
    CUBE3_COL_ID = 'id'              # Entity ID for output
    CUBE3_COL_NAME = 'title'
    CUBE3_COL_NICKNAME = 'nicknames'
    CUBE3_COL_STATUS = 'status'
    CUBE3_COL_ID_NUMBER = 'id_number'

    # ----- cube4 (Script 1 Output) Columns -----
    # Script 1 outputs: call_id, phone, original_id (links to cube3.~apak_id)
    CUBE4_COL_CALL_ID = 'call_id'
    CUBE4_COL_PHONE = 'phone'
    CUBE4_COL_ORIGINAL_ID = 'original_id'
    
    # ----- Output Columns (24 total) -----
    COL_FAKE_ID = 'fake_id'
    COL_ORIGINAL_ID = 'original_call_id'
    COL_NUM_A_OUT = 'num_A'
    COL_NUM_B_OUT = 'num_B'
    COL_DATE_OUT = 'date'
    
    # Extraction columns
    COL_NAME_ROLE_A = 'name+role_A'
    COL_NAME_ROLE_B = 'name+role_B'
    COL_NAME_CLEANED_A = 'name_cleaned_A'
    COL_NAME_CLEANED_B = 'name_cleaned_B'
    COL_ROLE_A = 'role_A'
    COL_ROLE_B = 'role_B'
    COL_EXTRACTION_METHOD = 'extraction_method'
    COL_STATUS = 'extraction_status'
    COL_METADATA = 'metadata'  # Combined: notes, swap_reason, br_found, had_swap
    
    # Verified entity columns (from cube3/cube4 integration)
    COL_VERIFIED_NAME_A = 'verified_name_A'
    COL_VERIFIED_NAME_B = 'verified_name_B'
    COL_VERIFIED_NICKNAME_A = 'verified_nickname_A'
    COL_VERIFIED_NICKNAME_B = 'verified_nickname_B'
    COL_VERIFIED_ENTITY_ID_A = 'verified_entity_id_A'
    COL_VERIFIED_ENTITY_ID_B = 'verified_entity_id_B'
    COL_VERIFIED_STATUS_A = 'verified_status_A'
    COL_VERIFIED_STATUS_B = 'verified_status_B'
    COL_VERIFIED_ID_NUMBER_A = 'verified_id_number_A'
    COL_VERIFIED_ID_NUMBER_B = 'verified_id_number_B'
    
    # Debug settings
    DEBUG_SWAP_CONSISTENCY = True  # Enable swap consistency checking

    # Role keywords for Stage 2 separation
    # v2.9.0: All abbreviations include: ", ״, ', ׳, and bare (no-quote) forms
    # ה-prefix matching is handled automatically in separate_name_from_role()
    ROLE_KEYWORDS = [
        # Command positions (with all quote variants + bare forms)
        'מפקד', 'מפקדת',
        'מג"ד', 'מג״ד', "מג'ד", 'מג׳ד', 'מגד',
        'מ"פ', 'מ״פ', "מ'פ", 'מ׳פ', 'מפ',
        'אג"ד', 'אג״ד', "אג'ד", 'אג׳ד', 'אגד',
        'סמג"ד', 'סמג״ד', "סמג'ד", 'סמג׳ד', 'סמגד',
        'מ"מ', 'מ״מ', "מ'מ", 'מ׳מ', 'ממ',
        'סגן',
        'אנ"ד', 'אנ״ד', "אנ'ד", 'אנ׳ד', 'אנד',
        # Note: רחל bare form intentionally OMITTED - רחל is a common female name
        'רח"ל', 'רח״ל', "רח'ל", 'רח׳ל',
        
        # Roles and positions
        'אחראי', 'אחראית',
        'קצין', 'קצינה',
        'מנהל', 'מנהלת',
        'מוביל', 'מובילה',
        'ס"ק', 'ס״ק', "ס'ק", 'ס׳ק', 'סק',
        'סמ"פ', 'סמ״פ', "סמ'פ", 'סמ׳פ', 'סמפ',
        
        # Personnel types
        'חייל',
        'פעיל', 'פעילה',
        'שומר',
        
        # Family/personal
        'אשתו', 'אישתו',
        'בנו',
        'בתו',
        
        # Additional roles (v2.8.1)
        'רל"ש', 'רל״ש', "רל'ש", 'רל׳ש', 'רלש',
        'רל"שו', 'רל״שו', "רל'שו", 'רל׳שו', 'רלשו',
        'סייען', 'סייענית',
        'מחליף', 'מחליפו', 'מחליפה',

        # Additional roles (v2.8.2) - from user feedback
        'רמ"ד', 'רמ״ד', "רמ'ד", 'רמ׳ד', 'רמד',  # Deputy commander
        'תצפיות',  # Observations - metadata suffix

        # Additional roles (v2.8.3) - for multi-person handling
        'מנהלן',  # Battalion manager (מנהל + ן suffix)
        'ממלא מקום', 'ממלא',  # Acting/substitute/fill-in

        # Additional roles (v2.8.4)
        'אח"ט', 'אח״ט', "אח'ט", 'אח׳ט', 'אחט',  # Logistics officer

        # v2.8.5: Note - bare forms like מגד are now included above
        # The rebuild logic converts 'מג ד' → 'מג״ד' before role separation
        
        # v2.8.7: Additional role keywords
        'מאייש',  # Manning/staffing

        # v2.9.0: New role keywords (with all quote variants + no-quote)
        # Each abbreviation has: ", ״, ', ׳, and bare (no quote) forms
        # ה-prefix matching is handled automatically in separate_name_from_role()
        
        # ראש (head/chief) - no quotes needed
        'ראש',
        
        # לשעבר (former) - no quotes needed  
        'לשעבר',
        
        # סמח״ט (deputy brigade commander)
        'סמח"ט', 'סמח״ט', "סמח'ט", 'סמח׳ט', 'סמחט',
        
        # סאג״ד (deputy battalion commander)
        'סאג"ד', 'סאג״ד', "סאג'ד", 'סאג׳ד', 'סאגד',
        
        # מח״ט (brigade commander)
        'מח"ט', 'מח״ט', "מח'ט", 'מח׳ט', 'מחט',
        
        # מפקצ (sector commander)
        'מפקצ',
    ]

    # Metadata phrases for Stage 2 separation (truncate from these)
    METADATA_PHRASES = [
        'מטעם',
        # v2.8.7: Additional metadata truncation triggers
        'כיום',       # "currently" - what follows is metadata
        'DTMF',       # Technical marker
        'שיח קודם',   # "previous conversation"
    ]
    
    # Header markers
    HEADER_START_MARKER = 'מקור'
    HEADER_END_MARKER = 'מזהה:'
    
    # Side markers
    SIDE_A_MARKER = re.compile(r'א\s*-', re.UNICODE)
    SIDE_B_MARKER = re.compile(r'ב\s*-', re.UNICODE)
    
    # BR Pattern for detection (MUST have geresh marks - no bare בר)
    BR_PATTERN = re.compile(
        r'(?:[-\s]*)?ב[״׳"\']ר',
        re.UNICODE
    )

    # Extended BR identification patterns
    BR_EXTENDED_PATTERNS = re.compile(
        r'(?:'
            r'על\s+פי|'
            r'ע[״׳"\']פ|'
            r'השוואת\s+עפ[״׳"\']?ק|'
            r'השוואת\s+עפק|'
            r'זוהה\s+עפ[״׳"\']?ק|'
            r'זוהה\s+עפק|'
            r'(?:ב)?זיהוי\s+דובר'
        r')',
        re.UNICODE
    )

    # SCHEMA Pattern
    SCHEMA_PATTERN = re.compile(
        r'\(\s*-?\s*ב[״׳"\']ר\s*\)?\s*/\s*מספר\s+\d+',
        re.UNICODE
    )
    
    # PRIORITY 1: Delimiter Pattern
    PATTERN_DELIMITER = re.compile(
        r'.*[/\\]+\s*(.+)$',
        re.UNICODE | re.DOTALL
    )
    
    # Pattern for text after final closing parenthesis
    PATTERN_AFTER_PAREN = re.compile(
        r'\)\s*([^()]+?)\s*$',
        re.UNICODE
    )
    
    # Cleaning patterns
    CLEANING_PATTERNS = {
        'numbers_long': re.compile(r'\b\d{4,}\b', re.UNICODE),
        'br_markers': re.compile(r'[-\s]*ב[״׳"\']ר[-\s]*', re.UNICODE),
        'br_extended_markers': re.compile(
            r'[-,:\s]*(?:'
                r'על\s+פי|'
                r'ע[״׳"\']פ|'
                r'השוואת\s+עפ[״׳"\']?ק|'
                r'השוואת\s+עפק|'
                r'זוהה\s+עפ[״׳"\']?ק|'
                r'זוהה\s+עפק|'
                r'(?:ב)?זיהוי\s+דובר'
            r')[-,:\s]*',
            re.UNICODE
        ),
        'hyphen_br': re.compile(r'-\s*ב[״׳"\']ר', re.UNICODE),
        'parentheses': re.compile(r'[()]', re.UNICODE),
        'leading_honorifics': re.compile(r'^(?:חאג\'|אלחאג\'|דוקטור)\s+', re.UNICODE),
        'verification': re.compile(
            r'(?:זוהה|מזוהה|הזדהה)\s+'
            r'(?:גם\s+)?'
            r'(?:'
                r'(?:ע[״׳"\']פ|עפ[״׳"\']י|על\s+פי)\s+(?:השוואת\s+)?|'
                r'על\s+ידי\s+(?:השוואת\s+)?|'
                r'לפי\s+(?:השוואת\s+)?'
            r')?'
            r'(?:עפ[״׳"\']?ק|עפק)',
            re.UNICODE
        ),
        'verification_voice': re.compile(
            r'(?:'
                r'(?:זוהה|מזוהה|הזדהה|זיהוי)\s+(?:גם\s+)?'
                r'(?:על\s+(?:פי|ידי)\s+)?'
            r'|'
                r'ע[״׳"\']פ\s+'
            r')'
            r'קול',
            re.UNICODE
        ),
        'verification_negative': re.compile(
            r'לאחר\s+השוואה?\s*(?:עפ[״׳"\']ק)?\s*(?:לא\s+מדובר|אין\s+מדובר|זה\s+לא)[^,)]*',
            re.UNICODE
        ),
        'verification_simple': re.compile(
            r',?\s*(?:זוהה|מזוהה|הזדהה)\s*$',
            re.UNICODE
        ),
        # v2.8.2: Additional cleaning patterns
        'percentage_verification': re.compile(
            r'[,\s]*\d+%\s*(?:עפ[״׳"\']?[קי])?',
            re.UNICODE
        ),
        'verification_partial': re.compile(
            r'[,\s]*(?:זוהה|מזוהה|הזדהה)\s+ע[״׳"\']י',
            re.UNICODE
        ),
        # v2.8.2: Remove ככה״נ (כך הוא נקרא - "this is what he's called")
        'kahan_marker': re.compile(
            r'[,\s]*ככה[״׳"\']נ[,\s]*',
            re.UNICODE
        ),
        'leading_junk': re.compile(r'^[\s,،、:)\-}\]]+|^[/\\]+\s*', re.UNICODE),
        'trailing_br_punctuation': re.compile(r'[\s,:\-]+$', re.UNICODE),
        'double_spaces': re.compile(r'\s+', re.UNICODE),
        'trailing_punctuation': re.compile(r'[\s,،、]+$', re.UNICODE),
        'double_commas': re.compile(r',\s*,', re.UNICODE),
        # v2.8.7: Updated junk_tokens - handles space and comma separators in עפ״ק
        # v2.9.0: Added removers: ראו, לפי, בהשוואת, ע״י (with all quote variants)
        # Matches tokens at word boundaries and removes them
        'junk_tokens': re.compile(
            r'(^|[\s,،、:;\-–—־()])'
            r'('
                r'השוואת|'
                r'בהשוואת|'  # v2.9.0
                r'עפ[״׳"\'\u05F3\u05F4\s,]?ק|'
                r'ע[״׳"\'\u05F3\u05F4]?י|'  # v2.9.0: ע״י, עי, ע׳י, etc.
                r'זוהה|'
                r'עולה|'
                r'ראו|'   # v2.9.0
                r'לפי'    # v2.9.0
            r')'
            r'(?=[\s,،、:;\-–—־()]|$)',
            re.UNICODE
        ),
        # v2.8.7: Updated BR pattern - handles ANY combination of space/comma/dash before ב״ר
        # Matches: - ב״ר, , - ב״ר, -ב״ר, ,- ב״ר, etc.
        'br_all_variants': re.compile(
            r'[\s,،、:;\-–—־]*ב[״׳"\'\u05F3\u05F4]ר[\s,،、:;\-–—־]*',
            re.UNICODE
        ),
        # v2.8.7: Final edge cleanup - catches any remaining punctuation at boundaries
        'edge_punctuation': re.compile(
            r'^[\s,،、:;\-–—־()\[\]{}"\'״׳]+|[\s,،、:;\-–—־()\[\]{}"\'״׳]+$',
            re.UNICODE
        ),
    }
    
    MAX_HEADER_LINES = 10
    EMPTY_PLACEHOLDER = ''
    
    # Status codes
    STATUS_SUCCESS = 'success'
    STATUS_PARTIAL = 'partial'
    STATUS_FAILED = 'failed'
    STATUS_EMPTY = 'empty'


# ============================================================================
# EARLY QUOTE NORMALIZATION - v2.8.9
# ============================================================================

def normalize_double_geresh(text: str) -> str:
    """
    v2.8.9: Early normalization of double-geresh variants to gershayim.
    
    Converts:
    - ׳׳ (two Hebrew geresh, U+05F3 × 2) → ״ (gershayim, U+05F4)
    - '' (two ASCII apostrophes) → ״
    
    This MUST run BEFORE pattern matching, because patterns like [״׳"\']
    use character classes that match single characters, not two-char sequences.
    
    Example: ב׳׳ר won't match ב[״׳"\']ר, but after this normalization,
    it becomes ב״ר which matches correctly.
    """
    if not text:
        return text
    text = text.replace('׳׳', '״')  # Two Hebrew geresh → gershayim
    text = text.replace("''", '״')  # Two ASCII apostrophes → gershayim
    return text


# ============================================================================
# BLMZ DETECTION - v2.8.7
# ============================================================================

# BLMZ Pattern: matches בלמז as a WORD anywhere in text
# Word boundary = start/end of string OR non-Hebrew character
_BLMZ_PATTERN = re.compile(
    r'(?:^|[^א-ת])'      # Word boundary before: start OR non-Hebrew
    r'בלמ'               # Root
    r'[\s״׳"\']?'        # Optional separator (space or any quote type)
    r'ז'                 # Mandatory ז
    r'(?:ית)?'           # Optional feminine suffix
    r'(?=$|[^א-ת])',     # Word boundary after: end OR non-Hebrew (lookahead)
    re.UNICODE
)

# Speaker placeholder: דובר/דוברת א/ב as a WORD anywhere
_SPEAKER_PLACEHOLDER_PATTERN = re.compile(
    r'(?:^|[^א-ת])'      # Word boundary before
    r'דובר(?:ת)?'        # דובר or דוברת
    r'\s*'               # Optional whitespace
    r'[אב]'              # Side marker
    r'[״׳"\']?'          # Optional trailing quote
    r'(?=$|[^א-ת])',     # Word boundary after (lookahead)
    re.UNICODE
)


def is_blmz(text: str) -> bool:
    """
    Check if text contains an unidentified speaker marker ANYWHERE.
    
    Returns True if ANY of these appear as a word:
    - בלמ״ז, בלמז, בלמ ז, בלמ'ז, בלמ׳ז
    - בלמזית, בלמ״זית, בלמ זית
    - דובר א, דובר א׳, דוברת ב״
    
    Word boundary prevents false matches like בלמזי (different word).
    """
    if not text:
        return False
    
    stripped = text.strip()
    if not stripped:
        return False
    
    # Search for BLMZ anywhere as a word
    if _BLMZ_PATTERN.search(stripped):
        return True
    
    # Search for speaker placeholder anywhere as a word
    if _SPEAKER_PLACEHOLDER_PATTERN.search(stripped):
        return True
    
    return False


def normalize_blmz(text: str) -> Tuple[str, bool]:
    """
    If text contains ANY form of BLMZ anywhere, return canonical בלמ״ז.
    
    Returns: (result, was_normalized)
    - If BLMZ found: ('בלמ״ז', True)
    - Otherwise: (original_text, False)
    """
    if is_blmz(text):
        return 'בלמ״ז', True
    return text, False


# ============================================================================
# EXTRACTION RESULT
# ============================================================================

@dataclass
class ExtractionResult:
    """Result with Stage 1 (name+role), and Stage 2 (separated) extractions"""
    raw_a: str
    raw_b: str
    name_role_a: str = ''
    name_role_b: str = ''
    name_cleaned_a: str = ''
    name_cleaned_b: str = ''
    role_a: str = ''
    role_b: str = ''
    had_swap: bool = False
    swap_reason: str = ''
    extraction_method: str = ''
    br_found: bool = False
    status: str = ''
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self, fake_id: str, original_id: str, num_a: str, num_b: str,
                date: str) -> dict:
        """Convert to dictionary with streamlined columns (v2.8.0)"""
        # Combine metadata into single field
        metadata_parts = []
        if self.had_swap:
            metadata_parts.append(f"swap:{self.swap_reason}")
        if self.br_found:
            metadata_parts.append("br_found")
        if self.notes:
            metadata_parts.append(' | '.join(self.notes))
        
        return {
            Config.COL_FAKE_ID: fake_id,
            Config.COL_ORIGINAL_ID: original_id,
            Config.COL_NUM_A_OUT: num_a,
            Config.COL_NUM_B_OUT: num_b,
            Config.COL_DATE_OUT: date,
            Config.COL_NAME_ROLE_A: self.name_role_a,
            Config.COL_NAME_ROLE_B: self.name_role_b,
            Config.COL_NAME_CLEANED_A: self.name_cleaned_a,
            Config.COL_NAME_CLEANED_B: self.name_cleaned_b,
            Config.COL_ROLE_A: self.role_a,
            Config.COL_ROLE_B: self.role_b,
            Config.COL_EXTRACTION_METHOD: self.extraction_method,
            Config.COL_STATUS: self.status,
            Config.COL_METADATA: ' | '.join(metadata_parts) if metadata_parts else '',
            # Verified columns - will be filled later by integrate_verified_entities
            Config.COL_VERIFIED_NAME_A: '',
            Config.COL_VERIFIED_NAME_B: '',
            Config.COL_VERIFIED_NICKNAME_A: '',
            Config.COL_VERIFIED_NICKNAME_B: '',
            Config.COL_VERIFIED_ENTITY_ID_A: '',
            Config.COL_VERIFIED_ENTITY_ID_B: '',
            Config.COL_VERIFIED_STATUS_A: '',
            Config.COL_VERIFIED_STATUS_B: '',
            Config.COL_VERIFIED_ID_NUMBER_A: '',
            Config.COL_VERIFIED_ID_NUMBER_B: '',
        }


# ============================================================================
# PRIORITY 1: DELIMITER EXTRACTION
# ============================================================================

def extract_with_delimiter(line: str) -> Tuple[Optional[str], str]:
    r"""Extract using delimiter at end of line (/, //, \, \\)"""
    if not line or not line.strip():
        return None, 'empty_line'
    
    match = Config.PATTERN_DELIMITER.search(line)
    
    if match:
        content = match.group(1).strip()
        if content:
            delimiter_match = re.search(r'([/\\]+)[^/\\]*$', line)
            if delimiter_match:
                delimiter = delimiter_match.group(1)
                return content, f'delimiter_{delimiter}'
            return content, 'delimiter'
    
    return None, 'no_delimiter'


# ============================================================================
# PRIORITY 2: BR PATTERN EXTRACTION
# ============================================================================

def find_br_patterns(text: str) -> List[Tuple[int, str]]:
    """Find all BR pattern positions and variations in text"""
    patterns = []
    for match in Config.BR_PATTERN.finditer(text):
        patterns.append((match.start(), match.group()))
    return patterns


def extract_with_br_pattern(line: str) -> Tuple[Optional[str], str]:
    """Extract text using BR pattern rules"""
    if not line or not line.strip():
        return None, 'empty_line'
    
    br_patterns = find_br_patterns(line)
    
    if not br_patterns:
        return None, 'no_br_pattern'
    
    last_br_pos, last_br_text = br_patterns[-1]
    
    if len(line) - last_br_pos < 20:
        text_before_br = line[:last_br_pos]
        open_paren = text_before_br.rfind('(')
        
        if open_paren != -1:
            content_in_paren = text_before_br[open_paren + 1:].strip()
            text_before_paren = text_before_br[:open_paren].strip()
            
            if text_before_paren:
                name_pattern = re.compile(
                    r'([א-ת][א-ת\'\'\-]{2,}(?:\s+[א-ת][א-ת\'\'\-]+)*)\s*$', 
                    re.UNICODE
                )
                name_match = name_pattern.search(text_before_paren)
                
                if name_match:
                    prefix_name = name_match.group(1).strip()
                    if len(prefix_name) > 3 and not prefix_name.startswith('א-') and not prefix_name.startswith('ב-'):
                        combined = f"{prefix_name} ({content_in_paren})"
                        return combined, 'br_pattern_with_kunya'
            
            if content_in_paren:
                return content_in_paren, 'br_pattern_parentheses'
        
        delim_match = re.search(r'([/\\]+)\s*([^/\\]+?)\s*(?:[-\s]*ב[״"\']ר)', line)
        if delim_match:
            extracted = delim_match.group(2).strip()
            if extracted:
                return extracted, 'br_pattern_delimiter'
    
    text_after_br = line[last_br_pos + len(last_br_text):].strip()
    text_after_br = text_after_br.rstrip(')')
    
    if text_after_br:
        text_after_br = text_after_br.lstrip('),،、( ')
        if text_after_br:
            return text_after_br, 'br_fallback_to_end'
    
    return None, 'br_found_but_no_content'


# ============================================================================
# PRIORITY 3: FALLBACK EXTRACTION
# ============================================================================

def extract_fallback(line: str) -> Tuple[Optional[str], str]:
    """Fallback extraction when delimiter and BR patterns fail"""
    if not line or not line.strip():
        return None, 'empty_line'
    
    match = Config.PATTERN_AFTER_PAREN.search(line)
    if match:
        content = match.group(1).strip()
        if content and len(content) > 2 and not content.startswith('-'):
            return content, 'fallback_after_paren'
    
    return None, 'no_fallback'


# ============================================================================
# EXTRACTION ORCHESTRATION
# ============================================================================

def extract_raw_from_line(line: str) -> Tuple[Optional[str], str, str]:
    """Orchestrate extraction with priority hierarchy"""
    if not line or not line.strip():
        return None, 'none', 'empty_line'
    
    schema_match = Config.SCHEMA_PATTERN.search(line)
    
    if schema_match:
        cleaned_line = line[:schema_match.start()] + line[schema_match.end():]
        
        delim_result, delim_note = extract_with_delimiter(cleaned_line)
        if delim_result:
            return delim_result, 'delimiter', f'schema_removed:{delim_note}'
        
        fallback_result, fallback_note = extract_fallback(cleaned_line)
        if fallback_result:
            return fallback_result, 'fallback', f'schema_removed:{fallback_note}'
        
        br_result, br_note = extract_with_br_pattern(cleaned_line)
        if br_result:
            return br_result, 'br_pattern', f'schema_removed:{br_note}'
        
        return None, 'none', f'schema_removed_but_failed'
    
    delim_result, delim_note = extract_with_delimiter(line)
    if delim_result:
        return delim_result, 'delimiter', delim_note
    
    fallback_result, fallback_note = extract_fallback(line)
    if fallback_result:
        return fallback_result, 'fallback', fallback_note
    
    br_result, br_note = extract_with_br_pattern(line)
    if br_result:
        return br_result, 'br_pattern', br_note
    
    return None, 'none', f'failed:all_methods'


# ============================================================================
# MANDATORY CLEANING AFTER EXTRACTION
# ============================================================================

def normalize_punctuation(text: str) -> str:
    r"""
    v2.8.8: Mission-safe quote normalization.
    
    Problem: In some data, ״ is used as word separator (מחמד״אלעג׳לה״זוהה״עפ״ק).
    Our cleaning patterns use \s+ so they won't match unless ״ becomes space.
    
    Solution:
    1. Normalize dashes and smart quotes
    2. Convert ALL ״ to spaces (treats as separator)
    3. Rebuild known abbreviations using word boundaries (to avoid corrupting names)
    
    Word boundary = abbreviation NOT followed by Hebrew letter.
    This prevents 'מג דוד' from becoming 'מג״דוד'.
    """
    if not text:
        return text

    # Step 1: Normalize dashes: en-dash, em-dash, Hebrew maqaf → ASCII hyphen
    text = re.sub(r'[–—־]', '-', text)

    # Step 2: Normalize smart quotes → gershayim first
    text = re.sub(r'[""„‟⁗]', '״', text)
    
    # Step 3: Normalize smart single quotes → geresh
    text = re.sub(r'[\u2018\u2019]', '׳', text)

    # Step 3b: Double-geresh → gershayim already done by normalize_double_geresh()
    # upstream in parse_content_for_names(). No need to repeat here.

    # Step 4: Convert ALL gershayim to spaces (mission-safe: no separator quotes survive)
    text = text.replace('״', ' ')

    # Step 5: Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Step 6: Rebuild known abbreviations using WORD BOUNDARIES
    # Only rebuild when NOT followed by Hebrew letter (prevents corrupting names)
    # Example: 'מג ד' at end → 'מג״ד', but 'מג דוד' stays 'מג דוד'
    ABBREVIATION_REBUILDS = [
        # Role abbreviations (from ROLE_KEYWORDS)
        # Note: ה-prefixed versions are included for abbreviations that commonly use them
        ('מג ד', 'מג״ד'), ('המג ד', 'המג״ד'),
        ('מ פ', 'מ״פ'), ('המ פ', 'המ״פ'),
        ('אג ד', 'אג״ד'), ('האג ד', 'האג״ד'),
        ('סמג ד', 'סמג״ד'), ('הסמג ד', 'הסמג״ד'),
        ('מ מ', 'מ״מ'), ('המ מ', 'המ״מ'),
        ('אנ ד', 'אנ״ד'), ('האנ ד', 'האנ״ד'),
        ('רח ל', 'רח״ל'), ('הרח ל', 'הרח״ל'),
        ('ס ק', 'ס״ק'), ('הס ק', 'הס״ק'),
        ('סמ פ', 'סמ״פ'), ('הסמ פ', 'הסמ״פ'),
        ('רל ש', 'רל״ש'), ('הרל ש', 'הרל״ש'),
        ('רל שו', 'רל״שו'), ('הרל שו', 'הרל״שו'),
        ('רמ ד', 'רמ״ד'), ('הרמ ד', 'הרמ״ד'),
        ('אח ט', 'אח״ט'), ('האח ט', 'האח״ט'),
        
        # v2.9.0: New role abbreviations (with ה-prefix variants)
        ('סמח ט', 'סמח״ט'), ('הסמח ט', 'הסמח״ט'),  # Deputy brigade commander
        ('סאג ד', 'סאג״ד'), ('הסאג ד', 'הסאג״ד'),  # Deputy battalion commander
        ('מח ט', 'מח״ט'), ('המח ט', 'המח״ט'),      # Brigade commander
        
        # Verification tokens
        ('ב ר', 'ב״ר'),
        ('ע פ', 'ע״פ'),
        ('עפ ק', 'עפ״ק'),
        ('עפ י', 'עפ״י'),
        ('ע י', 'ע״י'),
        
        # BLMZ marker (unidentified)
        ('בלמ ז', 'בלמ״ז'),
        ('בלמ זית', 'בלמ״זית'),
        
        # Other common abbreviations
        ('ככה נ', 'ככה״נ'),
    ]

    for spaced, abbrev in ABBREVIATION_REBUILDS:
        # Word boundary: not followed by Hebrew letter (א-ת)
        # This prevents 'מג דוד' from becoming 'מג״דוד'
        pattern = r'(?<![א-ת])' + re.escape(spaced) + r'(?![א-ת])'
        text = re.sub(pattern, abbrev, text)

    return text.strip()


def remove_numbers_from_name(text: str) -> str:
    """
    v2.8.4: Remove digit sequences from name (NOT from roles).
    Handles Western, Arabic-Indic, and Eastern Arabic-Indic digits.
    Also handles quoted number tokens like '60' or "340".
    """
    if not text:
        return text
    # Remove quoted number tokens like '60' or "340" (quotes immediately around digits only)
    # Also handles gershayim-quoted numbers after normalization: ״60״
    text = re.sub(r"['\"\u05F4][0-9٠-٩۰-۹]+['\"\u05F4]", '', text)
    # Remove all digit types: Western, Arabic-Indic, Eastern Arabic-Indic
    text = re.sub(r'[0-9٠-٩۰-۹]+', '', text)
    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    # Strip trailing punctuation
    text = text.rstrip(',،、:;-–—־').strip()
    return text


def clean_extracted_text(text: str) -> str:
    """Clean extracted text using ALL patterns - v2.8.8: handles ״ as separator"""
    if not text:
        return text

    # v2.8.7: Normalize punctuation (preserves quotes)
    cleaned = normalize_punctuation(text)

    # Remove patterns (order matters)
    cleaned = Config.CLEANING_PATTERNS['numbers_long'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['verification'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['verification_voice'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['verification_negative'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['verification_simple'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['percentage_verification'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['verification_partial'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['kahan_marker'].sub(' ', cleaned)
    
    # v2.8.7: junk_tokens BEFORE br_all_variants (so עפ״ק - ב״ר → ` - ב״ר` → ``)
    cleaned = Config.CLEANING_PATTERNS['junk_tokens'].sub(r'\1', cleaned)
    
    # BR removal patterns
    cleaned = Config.CLEANING_PATTERNS['br_markers'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['hyphen_br'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['br_extended_markers'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['br_all_variants'].sub(' ', cleaned)
    
    # Other cleanups
    cleaned = Config.CLEANING_PATTERNS['parentheses'].sub('', cleaned)
    cleaned = Config.CLEANING_PATTERNS['leading_honorifics'].sub('', cleaned)
    cleaned = Config.CLEANING_PATTERNS['leading_junk'].sub('', cleaned)
    cleaned = Config.CLEANING_PATTERNS['double_commas'].sub(',', cleaned)
    cleaned = re.sub(r'\s+,', ',', cleaned)
    cleaned = Config.CLEANING_PATTERNS['double_spaces'].sub(' ', cleaned)
    cleaned = Config.CLEANING_PATTERNS['trailing_punctuation'].sub('', cleaned)
    cleaned = Config.CLEANING_PATTERNS['trailing_br_punctuation'].sub('', cleaned)
    
    # v2.8.7: Final edge cleanup (catches any remaining junk at boundaries)
    cleaned = Config.CLEANING_PATTERNS['edge_punctuation'].sub('', cleaned)
    
    cleaned = cleaned.strip()

    # Remove wrapping quotes (all types)
    if cleaned and len(cleaned) > 2:
        quote_pairs = [('"', '"'), ("'", "'"), ('״', '״'), ('׳', '׳')]
        for open_q, close_q in quote_pairs:
            if cleaned[0] == open_q and cleaned[-1] == close_q:
                cleaned = cleaned[1:-1].strip()
                break

    return cleaned


# ============================================================================
# STAGE 2: ROLE SEPARATION
# ============================================================================

def separate_name_from_role(text: str) -> Tuple[str, str]:
    """Stage 2: Separate name from role/metadata - v2.9.0: auto-matches ה prefix"""
    if not text or not text.strip():
        return '', ''

    earliest_pos = len(text)
    separator_found = None

    # v2.8.4: Expanded boundary set to include colons, dashes, parentheses
    BOUNDARY = r'[\s,،、:;\-–—־()]'

    for phrase in Config.METADATA_PHRASES:
        pattern = r'(?:^|' + BOUNDARY + r')(' + re.escape(phrase) + r')(?:' + BOUNDARY + r'|$)'
        match = re.search(pattern, text, re.UNICODE)
        if match and match.start(1) < earliest_pos:
            earliest_pos = match.start(1)
            separator_found = 'metadata'

    for keyword in Config.ROLE_KEYWORDS:
        # v2.9.0: ה? prefix allows matching both מג״ד and המג״ד with same keyword
        pattern = r'(?:^|' + BOUNDARY + r')(ה?' + re.escape(keyword) + r')(?:' + BOUNDARY + r'|$)'
        match = re.search(pattern, text, re.UNICODE)
        if match and match.start(1) < earliest_pos:
            earliest_pos = match.start(1)
            separator_found = 'role'

    if separator_found is None:
        return text.strip(), ''

    name_part = text[:earliest_pos].strip()
    role_part = text[earliest_pos:].strip()
    # v2.8.4: Expanded rstrip to include colons, dashes, and opening parenthesis
    name_part = name_part.rstrip(',،、:;-–—־(').strip()

    return name_part, role_part


def process_multi_person_text(text: str) -> Tuple[str, str]:
    """
    Handle text with '+' separator indicating multiple people.
    Splits by '+', applies role separation to each segment, combines results.

    Returns (combined_names, combined_roles)

    Example: 'משה, מפקד + יוסי, מג״ד' → ('משה + יוסי', 'מפקד + מג״ד')
    """
    if not text or '+' not in text:
        return separate_name_from_role(text)

    # Split by '+' and process each segment
    segments = [s.strip() for s in text.split('+') if s.strip()]

    if len(segments) <= 1:
        return separate_name_from_role(text)

    names = []
    roles = []
    for segment in segments:
        # Apply role separation to each segment
        name, role = separate_name_from_role(segment)
        if name:
            names.append(name.strip())
        if role:
            roles.append(role.strip())

    combined_names = ' + '.join(names) if names else ''
    combined_roles = ' + '.join(roles) if roles else ''

    return combined_names, combined_roles


# ============================================================================
# VALIDATION
# ============================================================================

def validate_input(cubes: dict) -> Tuple[bool, str]:
    """Validate input cubes (cube1 and cube5 required, cube3/cube4 optional)"""
    if not isinstance(cubes, dict):
        return False, "Input must be dictionary"
    
    if Config.INPUT_CUBE not in cubes:
        return False, f"Missing required cube: {Config.INPUT_CUBE}"
    
    df = cubes[Config.INPUT_CUBE]
    if not isinstance(df, pd.DataFrame):
        return False, f"{Config.INPUT_CUBE} must be DataFrame"
    
    if df.empty:
        return False, f"{Config.INPUT_CUBE} is empty"
    
    if Config.COL_CONTENT not in df.columns:
        return False, f"Missing required column: {Config.COL_CONTENT}"
    
    if Config.INPUT_CUBE_2 not in cubes:
        return False, f"Missing required cube: {Config.INPUT_CUBE_2}"
    
    df2 = cubes[Config.INPUT_CUBE_2]
    if not isinstance(df2, pd.DataFrame):
        return False, f"{Config.INPUT_CUBE_2} must be DataFrame"
    
    if df2.empty:
        return False, f"{Config.INPUT_CUBE_2} is empty"
    
    if Config.COL_MAIN_NUMBER not in df2.columns:
        return False, f"Missing required column: {Config.COL_MAIN_NUMBER}"
    
    return True, ""


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate_by_latest_call(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Keep only the latest call for each call_id based on date."""
    if df.empty:
        return df, {'deduplication': 'skipped', 'reason': 'empty'}
    
    if Config.COL_CALL_ID not in df.columns:
        return df, {'deduplication': 'skipped', 'reason': 'no_call_id'}
    
    df_work = df.copy()
    df_work['_original_idx'] = range(len(df_work))
    
    def normalize_call_id(val):
        if pd.isna(val):
            return None
        try:
            normalized = str(val).strip()
            return normalized if normalized else None
        except:
            return None
    
    df_work['_normalized_call_id'] = df_work[Config.COL_CALL_ID].apply(normalize_call_id)
    
    mask_has_valid_id = df_work['_normalized_call_id'].notna()
    df_with_id = df_work[mask_has_valid_id].copy()
    df_no_id = df_work[~mask_has_valid_id].copy()
    
    if df_with_id.empty:
        return df, {'deduplication': 'skipped', 'reason': 'no_valid_ids'}
    
    def parse_date_safe(date_val):
        if pd.isna(date_val):
            return pd.NaT
        try:
            return pd.to_datetime(date_val)
        except:
            return pd.NaT
    
    if Config.COL_DATE in df_with_id.columns:
        df_with_id['_parsed_date'] = df_with_id[Config.COL_DATE].apply(parse_date_safe)
    else:
        df_with_id['_parsed_date'] = pd.NaT
    
    df_with_id_sorted = df_with_id.sort_values(
        by=['_parsed_date', '_original_idx'],
        ascending=[False, False],
        na_position='last'
    )
    
    df_dedup = df_with_id_sorted.drop_duplicates(
        subset=['_normalized_call_id'],
        keep='first'
    )
    
    if not df_no_id.empty:
        result_df = pd.concat([df_dedup, df_no_id], ignore_index=True)
    else:
        result_df = df_dedup.copy()
        result_df = result_df.reset_index(drop=True)
    
    result_df = result_df.sort_values('_original_idx')
    result_df = result_df.reset_index(drop=True)
    result_df = result_df.drop(columns=['_original_idx', '_normalized_call_id', '_parsed_date'], errors='ignore')
    
    stats = {
        'original_count': len(df),
        'deduplicated_count': len(result_df),
        'duplicates_removed': len(df) - len(result_df),
    }
    
    return result_df, stats


# ============================================================================
# PHONE NUMBER NORMALIZATION
# ============================================================================

def normalize_phone_number(phone: Any) -> str:
    """Normalize phone number to digits only"""
    if phone is None or (isinstance(phone, float) and pd.isna(phone)):
        return ''
    
    if isinstance(phone, (int, float, np.integer, np.floating)):
        phone_str = f"{int(phone)}"
    else:
        phone_str = str(phone).strip()
        # Handle float→string artifact: "972501234567.0" → "972501234567"
        # This happens when DataFrame phone columns are float64 (due to NaN values)
        # and str() conversion preserves the ".0" suffix.
        if phone_str.endswith('.0'):
            phone_str = phone_str[:-2]

    return re.sub(r'\D', '', phone_str)


def _normalize_israeli_phone(digits: str) -> str:
    """
    Normalize Israeli phone to canonical format (local 9-digit without leading 0).

    Handles:
    - International: 972501234567 → 501234567
    - Local with 0:  0501234567   → 501234567
    - Already clean: 501234567    → 501234567
    """
    if not digits:
        return ''

    # Remove 972/970 country code prefix if present
    if (digits.startswith('972') or digits.startswith('970')) and len(digits) > 10:
        digits = digits[3:]  # Remove '972' or '970'

    # Remove leading zero (local format)
    # Mobile: 0501234567 (10 digits) → 501234567
    # Landline: 021234567 (9 digits) → 21234567
    if digits.startswith('0') and len(digits) in (9, 10):
        digits = digits[1:]  # Remove leading '0'

    return digits


def normalize_phone_for_output(phone: Any) -> str:
    """Normalize phone to Israeli local format (0XXXXXXXXX) for pipeline output.

    Matches the Algorithm's normalize_phone() canonical form so there is
    ONE consistent format across the entire pipeline (Point → Algorithm → output).

    972501234567 → 0501234567
    970591234567 → 0591234567
    0501234567   → 0501234567
    501234567    → 0501234567
    """
    digits = normalize_phone_number(phone)
    if not digits:
        return ''
    # Strip international prefix, re-add local leading 0
    # Must match Algorithm's normalize_phone() exactly — no length guard
    if digits.startswith('972') or digits.startswith('970'):
        digits = '0' + digits[3:]
    # Bare 9-digit → add leading 0
    if len(digits) == 9 and not digits.startswith('0'):
        digits = '0' + digits
    return digits


def phone_numbers_match(phone1: str, phone2: str) -> bool:
    """
    Check if two phone numbers match after normalization.

    Handles Israeli phone format variations:
    - 972501234567 matches 0501234567 matches 501234567
    - Also handles leading zeros in general
    """
    norm1 = normalize_phone_number(phone1)
    norm2 = normalize_phone_number(phone2)

    if not norm1 or not norm2:
        return False

    # Direct match
    if norm1 == norm2:
        return True

    # Israeli format normalization (handles 972 prefix and leading 0)
    israeli1 = _normalize_israeli_phone(norm1)
    israeli2 = _normalize_israeli_phone(norm2)

    if israeli1 and israeli2 and israeli1 == israeli2:
        return True

    # Fallback: compare last 9 digits (handles any format variations)
    if len(norm1) >= 9 and len(norm2) >= 9:
        if norm1[-9:] == norm2[-9:]:
            return True

    # Final fallback: strip all leading zeros
    stripped1 = norm1.lstrip('0')
    stripped2 = norm2.lstrip('0')

    return stripped1 and stripped2 and stripped1 == stripped2


# ============================================================================
# SIDE NORMALIZATION
# ============================================================================

def should_swap_sides(num_a: str, num_b: str, main_number: str) -> Tuple[bool, str]:
    """Determine if sides should be swapped"""
    if not main_number or pd.isna(main_number):
        return False, 'no_main_number'
    
    main_norm = normalize_phone_number(main_number)
    if not main_norm:
        return False, 'invalid_main_number'
    
    if phone_numbers_match(num_a, main_number):
        return False, 'side_a_is_main'
    
    if phone_numbers_match(num_b, main_number):
        return True, 'side_b_is_main'
    
    return False, 'main_number_not_found'


def swap_extraction_result(result: ExtractionResult) -> ExtractionResult:
    """Swap all fields between Side A and Side B"""
    return ExtractionResult(
        raw_a=result.raw_b,
        raw_b=result.raw_a,
        name_role_a=result.name_role_b,
        name_role_b=result.name_role_a,
        name_cleaned_a=result.name_cleaned_b,
        name_cleaned_b=result.name_cleaned_a,
        role_a=result.role_b,
        role_b=result.role_a,
        had_swap=True,
        swap_reason=result.swap_reason,
        extraction_method=result.extraction_method,
        br_found=result.br_found,
        status=result.status,
        notes=result.notes + ['sides_swapped']
    )


# ============================================================================
# SWAP CONSISTENCY VALIDATION (DEBUG)
# ============================================================================

def validate_swap_consistency(
    call_id: str,
    should_swap: bool,
    orig_num_a: str,
    orig_num_b: str,
    orig_name_a: str,
    orig_name_b: str,
    final_num_a: str,
    final_num_b: str,
    final_name_a: str,
    final_name_b: str
) -> Tuple[bool, str]:
    """
    Validate that phone swap and name swap are consistent.
    Returns (is_consistent, debug_message)
    """
    # Determine if phones actually swapped
    phones_swapped = phone_numbers_match(final_num_a, orig_num_b) if orig_num_b else False
    phones_stayed = phone_numbers_match(final_num_a, orig_num_a) if orig_num_a else False

    # Determine if names actually swapped (handle empty strings)
    names_swapped = (final_name_a == orig_name_b) if (orig_name_b and final_name_a) else False
    names_stayed = (final_name_a == orig_name_a) if (orig_name_a and final_name_a) else False

    # Check consistency
    is_consistent = True
    issues = []

    if should_swap:
        # Swap was requested
        if orig_num_b and not phones_swapped:
            issues.append("PHONE_NOT_SWAPPED")
        if orig_name_b and not names_swapped:
            issues.append("NAME_NOT_SWAPPED")
    else:
        # No swap requested
        if orig_num_a and phones_swapped:
            issues.append("PHONE_SWAPPED_UNEXPECTEDLY")
        if orig_name_a and names_swapped and orig_name_a != orig_name_b:
            issues.append("NAME_SWAPPED_UNEXPECTEDLY")

    # Critical: phones and names should swap together
    if phones_swapped != names_swapped:
        # Only flag if both sides had values to swap
        if (orig_name_a or orig_name_b) and (orig_num_a or orig_num_b):
            issues.append("PHONE_NAME_MISMATCH")
            is_consistent = False

    if issues:
        # Truncate names for display
        def truncate(s, max_len=30):
            if not s:
                return ''
            return s[:max_len] + '...' if len(s) > max_len else s

        debug_msg = (
            f"[SWAP_DEBUG] call_id={call_id} | should_swap={should_swap} | "
            f"phones_swapped={phones_swapped} | names_swapped={names_swapped} | "
            f"issues={issues} | "
            f"orig_phones=({orig_num_a}, {orig_num_b}) | "
            f"final_phones=({final_num_a}, {final_num_b}) | "
            f"orig_names=({truncate(orig_name_a)}, {truncate(orig_name_b)}) | "
            f"final_names=({truncate(final_name_a)}, {truncate(final_name_b)})"
        )
        return is_consistent, debug_msg

    return True, ""


# ============================================================================
# HEADER REMOVAL
# ============================================================================

def remove_header_block(content: str) -> Tuple[str, bool]:
    """Remove header section if present"""
    if not content or not content.strip():
        return content, False
    
    lines = content.split('\n')
    header_start_idx = -1
    header_end_idx = -1
    
    for i, line in enumerate(lines[:Config.MAX_HEADER_LINES]):
        if Config.HEADER_START_MARKER in line and header_start_idx == -1:
            header_start_idx = i
        if Config.HEADER_END_MARKER in line and header_start_idx != -1:
            header_end_idx = i
            break
    
    if header_start_idx != -1 and header_end_idx != -1:
        cleaned_lines = lines[:header_start_idx] + lines[header_end_idx + 1:]
        return '\n'.join(cleaned_lines), True
    
    return content, False


# ============================================================================
# LINE FINDING
# ============================================================================

def find_side_line(content: str, marker_pattern: re.Pattern) -> Optional[str]:
    """Find line starting with given marker pattern"""
    if not content:
        return None
    
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if marker_pattern.match(stripped):
            return line
    
    return None


# ============================================================================
# CORE PROCESSING
# ============================================================================

def parse_content_for_names(content: str) -> ExtractionResult:
    """Parse IntelliItemContent with two-stage extraction - v2.8.9"""
    notes = []
    
    if not content or not isinstance(content, str) or not content.strip():
        return ExtractionResult(
            raw_a=Config.EMPTY_PLACEHOLDER,
            raw_b=Config.EMPTY_PLACEHOLDER,
            status=Config.STATUS_EMPTY,
            notes=['empty_or_invalid_content']
        )
    
    # v2.8.9: Early normalization - convert ׳׳ to ״ BEFORE pattern matching
    # This ensures patterns like ב[״׳"\']ר will match ב׳׳ר (after it becomes ב״ר)
    content = normalize_double_geresh(content)
    
    br_found = bool(Config.BR_PATTERN.search(content) or Config.BR_EXTENDED_PATTERNS.search(content))
    
    cleaned_content, header_found = remove_header_block(content)
    if header_found:
        notes.append('header_removed')
    
    # Find Side A line
    side_a_line = find_side_line(cleaned_content, Config.SIDE_A_MARKER)
    if side_a_line:
        raw_a, method_a, note_a = extract_raw_from_line(side_a_line)
        notes.append(f'side_a:{note_a}')
    else:
        raw_a = None
        method_a = 'none'
        notes.append('side_a:line_not_found')
    
    # Find Side B line
    side_b_line = find_side_line(cleaned_content, Config.SIDE_B_MARKER)
    if side_b_line:
        raw_b, method_b, note_b = extract_raw_from_line(side_b_line)
        notes.append(f'side_b:{note_b}')
    else:
        raw_b = None
        method_b = 'none'
        notes.append('side_b:line_not_found')
    
    # Determine extraction method
    if 'delimiter' in method_a or 'delimiter' in method_b:
        extraction_method = 'delimiter'
    elif 'br_pattern' in method_a or 'br_pattern' in method_b:
        extraction_method = 'br_pattern'
    elif 'fallback' in method_a or 'fallback' in method_b:
        extraction_method = 'fallback'
    else:
        extraction_method = 'none'
    
    # STAGE 1: Clean extracted text
    name_role_a = clean_extracted_text(raw_a) if raw_a else Config.EMPTY_PLACEHOLDER
    name_role_b = clean_extracted_text(raw_b) if raw_b else Config.EMPTY_PLACEHOLDER
    
    # STAGE 2: Separate name from role (v2.8.3: handles '+' multi-person)
    name_cleaned_a, role_a = process_multi_person_text(name_role_a)
    name_cleaned_b, role_b = process_multi_person_text(name_role_b)

    # Track if multi-person processing occurred
    if '+' in name_role_a and '+' in name_cleaned_a:
        notes.append('side_a:multi_person_processed')
    if '+' in name_role_b and '+' in name_cleaned_b:
        notes.append('side_b:multi_person_processed')
    
    # v2.8.1: Invalidate results containing משמש/משמשת (metadata, not names)
    INVALID_MARKERS = ['משמש', 'משמשת']
    for marker in INVALID_MARKERS:
        if marker in name_role_a:
            name_role_a = Config.EMPTY_PLACEHOLDER
            name_cleaned_a = ''
            role_a = ''
            notes.append('side_a:invalidated_meshammesh')
            break
    for marker in INVALID_MARKERS:
        if marker in name_role_b:
            name_role_b = Config.EMPTY_PLACEHOLDER
            name_cleaned_b = ''
            role_b = ''
            notes.append('side_b:invalidated_meshammesh')
            break

    # v2.8.4: Invalidate השלוחה (branch/extension, not a person)
    if 'השלוחה' in name_role_a:
        name_role_a = Config.EMPTY_PLACEHOLDER
        name_cleaned_a = ''
        role_a = ''
        notes.append('side_a:invalidated_hashluha')
    if 'השלוחה' in name_role_b:
        name_role_b = Config.EMPTY_PLACEHOLDER
        name_cleaned_b = ''
        role_b = ''
        notes.append('side_b:invalidated_hashluha')

    # v2.8.4: Remove numbers from names (NOT from roles)
    name_cleaned_a = remove_numbers_from_name(name_cleaned_a)
    name_cleaned_b = remove_numbers_from_name(name_cleaned_b)

    # v2.8.7: Normalize ALL BLMZ variations to canonical form
    # This replaces the old speaker placeholder logic
    name_cleaned_a, was_blmz_a = normalize_blmz(name_cleaned_a)
    name_cleaned_b, was_blmz_b = normalize_blmz(name_cleaned_b)
    
    if was_blmz_a:
        notes.append('side_a:blmz_normalized')
    if was_blmz_b:
        notes.append('side_b:blmz_normalized')

    if role_a:
        notes.append('side_a:role_separated')
    if role_b:
        notes.append('side_b:role_separated')
    
    # Determine status
    if raw_a and raw_b:
        status = Config.STATUS_SUCCESS
    elif raw_a or raw_b:
        status = Config.STATUS_PARTIAL
    else:
        status = Config.STATUS_FAILED
    
    final_raw_a = raw_a if raw_a else Config.EMPTY_PLACEHOLDER
    final_raw_b = raw_b if raw_b else Config.EMPTY_PLACEHOLDER
    
    return ExtractionResult(
        raw_a=final_raw_a,
        raw_b=final_raw_b,
        name_role_a=name_role_a,
        name_role_b=name_role_b,
        name_cleaned_a=name_cleaned_a,
        name_cleaned_b=name_cleaned_b,
        role_a=role_a,
        role_b=role_b,
        extraction_method=extraction_method,
        br_found=br_found,
        status=status,
        notes=notes
    )


def process_dataframe(df: pd.DataFrame, main_number: str) -> pd.DataFrame:
    """Process entire DataFrame with corrected extraction"""
    num_rows = len(df)
    if num_rows < 100:
        fake_id_format = '{:02d}'
    elif num_rows < 1000:
        fake_id_format = '{:03d}'
    elif num_rows < 10000:
        fake_id_format = '{:04d}'
    else:
        fake_id_format = '{:05d}'
    
    if Config.COL_CALL_ID in df.columns:
        original_ids = df[Config.COL_CALL_ID].tolist()
    else:
        original_ids = [str(i) for i in range(1, num_rows + 1)]
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            content = row[Config.COL_CONTENT]
            result = parse_content_for_names(content)

            # Get original phone numbers from input
            orig_num_a = str(row.get(Config.COL_NUM_A, '') or '')
            orig_num_b = str(row.get(Config.COL_NUM_B, '') or '')

            # Store original names BEFORE any swap
            orig_name_a = result.name_cleaned_a
            orig_name_b = result.name_cleaned_b

            # Initialize working variables
            num_a = orig_num_a
            num_b = orig_num_b

            should_swap, swap_reason = should_swap_sides(num_a, num_b, main_number)

            result.swap_reason = swap_reason
            result.notes.append(f'normalization:{swap_reason}')

            if should_swap:
                result = swap_extraction_result(result)
                # ALSO swap num_a/num_b so all "Side A" data is consistent
                num_a, num_b = num_b, num_a

            # DEBUG: Validate swap consistency
            if Config.DEBUG_SWAP_CONSISTENCY:
                original_id_debug = str(original_ids[len(results)]) if len(results) < len(original_ids) else 'unknown'
                is_consistent, debug_msg = validate_swap_consistency(
                    call_id=original_id_debug,
                    should_swap=should_swap,
                    orig_num_a=orig_num_a,
                    orig_num_b=orig_num_b,
                    orig_name_a=orig_name_a,
                    orig_name_b=orig_name_b,
                    final_num_a=num_a,
                    final_num_b=num_b,
                    final_name_a=result.name_cleaned_a,
                    final_name_b=result.name_cleaned_b
                )
                if debug_msg:
                    print(debug_msg)
                if not is_consistent:
                    result.notes.append('SWAP_INCONSISTENCY_DETECTED')

            row_num = idx + 1 if isinstance(idx, int) else len(results) + 1
            fake_id = fake_id_format.format(row_num)
            
            original_id = str(original_ids[len(results)])
            date = row.get(Config.COL_DATE, '')
            
            result_dict = result.to_dict(
                fake_id=fake_id,
                original_id=original_id,
                num_a=normalize_phone_for_output(num_a),
                num_b=normalize_phone_for_output(num_b),
                date=str(date)
            )
            
            results.append(result_dict)
            
        except Exception as e:
            row_num = len(results) + 1
            fake_id = fake_id_format.format(row_num)
            results.append({
                Config.COL_FAKE_ID: fake_id,
                Config.COL_ORIGINAL_ID: str(original_ids[len(results)]) if len(results) < len(original_ids) else '',
                Config.COL_STATUS: 'error',
                Config.COL_METADATA: f'exception:{str(e)[:100]}'
            })
    
    return pd.DataFrame(results)


# ============================================================================
# VERIFIED ENTITY INTEGRATION (NEW v2.8.0)
# ============================================================================

def _normalize_entity_id(entity_id: str) -> str:
    """
    Normalize entity ID for matching:
    - Handle NaN string values
    - Strip whitespace

    Examples:
        'nan' -> '' (empty, will be filtered)
        'e-1399fsr5' -> 'e-1399fsr5' (unchanged)
        '' -> '' (empty)
    """
    if pd.isna(entity_id):
        return ''

    s = str(entity_id).strip()

    # Handle NaN string representations
    if s.lower() in ('nan', 'none', 'null', ''):
        return ''

    return s


def integrate_verified_entities(result_df: pd.DataFrame, cube3: pd.DataFrame, cube4: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate verified entity names from cube3 (enhancement) and cube4 (Script 1).

    Flow:
    1. cube4 has: call_id, phone, original (Script 1 output - links calls to entities)
    2. cube3 has: id, title, nicknames, status, id_number (enhancement table - entity data)
    3. Join cube4 + cube3 on entity ID
    4. Create lookup by call_id -> list of (phone, name, nickname, entity_id, status, id_number)
    5. For each Point row, match by call_id, then determine side by comparing phone to num_A/num_B
    """
    print(f"\n[DEBUG] integrate_verified_entities called")
    print(f"[DEBUG] cube3 shape: {cube3.shape if cube3 is not None else 'None'}")
    print(f"[DEBUG] cube4 shape: {cube4.shape if cube4 is not None else 'None'}")
    print(f"[DEBUG] result_df shape: {result_df.shape}")

    if cube3 is None or cube3.empty or cube4 is None or cube4.empty:
        print(f"[DEBUG] EARLY EXIT: cube3 or cube4 is None/empty")
        return result_df

    print(f"[DEBUG] cube4 columns: {list(cube4.columns)}")
    print(f"[DEBUG] cube3 columns: {list(cube3.columns)}")

    # Verify required columns exist in cube4 (Script 1 output)
    cube4_call_id_col = Config.CUBE4_COL_CALL_ID
    cube4_phone_col = Config.CUBE4_COL_PHONE
    cube4_original_id_col = Config.CUBE4_COL_ORIGINAL_ID

    missing_cube4 = [c for c in [cube4_call_id_col, cube4_phone_col, cube4_original_id_col] if c not in cube4.columns]
    if missing_cube4:
        print(f"[DEBUG] EARLY EXIT: Missing columns in cube4: {missing_cube4}")
        return result_df

    print(f"[DEBUG] cube4 cols: call_id='{cube4_call_id_col}', phone='{cube4_phone_col}', original_id='{cube4_original_id_col}'")

    # Verify required columns exist in cube3 (enhancement table)
    cube3_apak_id_col = Config.CUBE3_COL_APAK_ID  # Join key
    cube3_id_col = Config.CUBE3_COL_ID             # Entity ID for output
    cube3_name_col = Config.CUBE3_COL_NAME
    cube3_nickname_col = Config.CUBE3_COL_NICKNAME
    cube3_status_col = Config.CUBE3_COL_STATUS
    cube3_id_number_col = Config.CUBE3_COL_ID_NUMBER

    missing_cube3 = [c for c in [cube3_apak_id_col, cube3_id_col, cube3_name_col] if c not in cube3.columns]
    if missing_cube3:
        print(f"[DEBUG] EARLY EXIT: Missing columns in cube3: {missing_cube3}")
        return result_df

    print(f"[DEBUG] cube3 cols: apak_id='{cube3_apak_id_col}', id='{cube3_id_col}', name='{cube3_name_col}', nickname='{cube3_nickname_col}', status='{cube3_status_col}', id_number='{cube3_id_number_col}'")

    # Step 1: Prepare cube4 (Script 1 output)
    # cube4.original links to cube3.~apak_id (join key)
    cube4_clean = cube4[[cube4_call_id_col, cube4_phone_col, cube4_original_id_col]].copy()
    cube4_clean.columns = ['call_id', 'phone', 'join_key']
    cube4_clean['call_id'] = cube4_clean['call_id'].astype(str).str.strip()
    cube4_clean['phone'] = cube4_clean['phone'].apply(normalize_phone_for_output)
    # Normalize join_key (handles NaN, leading zeros, prefixes)
    cube4_clean['join_key'] = cube4_clean['join_key'].apply(_normalize_entity_id)

    # Filter out rows with empty join_key (from NaN values)
    cube4_before = len(cube4_clean)
    cube4_clean = cube4_clean[cube4_clean['join_key'] != '']
    cube4_filtered = cube4_before - len(cube4_clean)
    if cube4_filtered > 0:
        print(f"[DEBUG] cube4: Filtered out {cube4_filtered} rows with empty/NaN join_key")

    print(f"[DEBUG] cube4_clean sample (first 3 rows):")
    print(cube4_clean.head(3).to_string())

    # Step 2: Prepare cube3 (enhancement table)
    # - ~apak_id: join key (links to cube4.original)
    # - id: verified entity ID (for output)
    # - title: name
    # - nicknames: nicknames
    # - status: entity status
    # - id_number: entity id_number
    cube3_cols = [cube3_apak_id_col, cube3_id_col, cube3_name_col]
    col_names = ['join_key', 'verified_entity_id', 'verified_name']

    has_nickname = cube3_nickname_col in cube3.columns
    if has_nickname:
        cube3_cols.append(cube3_nickname_col)
        col_names.append('verified_nickname')

    has_status = cube3_status_col in cube3.columns
    if has_status:
        cube3_cols.append(cube3_status_col)
        col_names.append('verified_status')

    has_id_number = cube3_id_number_col in cube3.columns
    if has_id_number:
        cube3_cols.append(cube3_id_number_col)
        col_names.append('verified_id_number')

    cube3_clean = cube3[cube3_cols].copy()
    cube3_clean.columns = col_names

    if not has_nickname:
        cube3_clean['verified_nickname'] = ''
    if not has_status:
        cube3_clean['verified_status'] = ''
    if not has_id_number:
        cube3_clean['verified_id_number'] = ''

    # Normalize join_key (handles NaN, leading zeros, prefixes)
    cube3_clean['join_key'] = cube3_clean['join_key'].apply(_normalize_entity_id)

    # Filter out rows with empty join_key (from NaN values)
    cube3_before = len(cube3_clean)
    cube3_clean = cube3_clean[cube3_clean['join_key'] != '']
    cube3_filtered = cube3_before - len(cube3_clean)
    if cube3_filtered > 0:
        print(f"[DEBUG] cube3: Filtered out {cube3_filtered} rows with empty/NaN join_key")

    print(f"[DEBUG] cube3_clean sample (first 3 rows):")
    print(cube3_clean.head(3).to_string())
    
    # DIAGNOSTIC: Check verified_entity_id in cube3
    print(f"[DIAG] cube3_clean columns: {list(cube3_clean.columns)}")
    print(f"[DIAG] cube3_clean dtypes: {cube3_clean.dtypes.to_dict()}")
    is_valid_eid = cube3_clean['verified_entity_id'].apply(
        lambda x: x is not None and str(x).strip() not in ('', 'nan', 'None', 'NaN')
    )
    non_empty_eids = is_valid_eid.sum()
    print(f"[DIAG] cube3 verified_entity_id non-empty: {non_empty_eids}/{len(cube3_clean)}")
    if non_empty_eids > 0:
        sample_eids = cube3_clean[is_valid_eid]['verified_entity_id'].head(5).tolist()
        print(f"[DIAG] Sample cube3 entity_id values: {sample_eids}")
    else:
        # Check raw values before filtering
        print(f"[DIAG] cube3 verified_entity_id raw sample: {cube3_clean['verified_entity_id'].head(10).tolist()}")

    # Step 3: Join cube4 + cube3 on join_key (~apak_id)
    # This connects the Script 1 rows to the enhancement table entries
    enriched = cube4_clean.merge(cube3_clean, on='join_key', how='left')

    # Count how many got matched
    matched_count = enriched['verified_name'].notna().sum()
    print(f"[DEBUG] After join: {len(enriched)} rows, {matched_count} with names from enhancer")

    if matched_count == 0:
        print(f"[DEBUG] WARNING: No join_key matches between cube4 and cube3!")
        print(f"[DEBUG] cube4 join_keys sample: {cube4_clean['join_key'].head(5).tolist()}")
        print(f"[DEBUG] cube3 join_keys sample: {cube3_clean['join_key'].head(5).tolist()}")

    # Step 4: Create lookup by call_id -> list of (phone, name, nickname, verified_entity_id, status, id_number)
    # Each call_id may have multiple entries (one per side that had grade=100)
    lookup: Dict[str, List[Tuple[str, str, str, str, str, str]]] = {}
    entity_ids_found = 0
    entity_ids_empty = 0

    for _, row in enriched.iterrows():
        call_id = str(row['call_id']).strip()
        phone = str(row['phone']).strip()
        def _safe_str(val):
            return str(val) if pd.notna(val) else ''

        name = _safe_str(row.get('verified_name'))
        nickname = _safe_str(row.get('verified_nickname'))
        status = _safe_str(row.get('verified_status'))
        id_number = _safe_str(row.get('verified_id_number'))
        verified_entity_id = _safe_str(row.get('verified_entity_id'))

        if verified_entity_id and verified_entity_id.strip() and verified_entity_id.lower() not in ('nan', 'none'):
            entity_ids_found += 1
        else:
            entity_ids_empty += 1
            verified_entity_id = ''  # Normalize to empty string

        if name:  # Only add if we have a name from enhancer
            if call_id not in lookup:
                lookup[call_id] = []
            lookup[call_id].append((phone, name, nickname, verified_entity_id, status, id_number))

    print(f"[DEBUG] Lookup has {len(lookup)} call_ids with verified names")
    print(f"[DIAG] Entity IDs found: {entity_ids_found}, empty: {entity_ids_empty}")
    
    if lookup:
        sample_key = list(lookup.keys())[0]
        sample_entry = lookup[sample_key]
        print(f"[DEBUG] Sample lookup entry: '{sample_key}' -> {sample_entry}")
        # Show entity_id specifically
        print(f"[DIAG] Sample entity_id in lookup: '{sample_entry[0][3] if sample_entry else 'N/A'}'")
    
    # Count entries with non-empty entity_id
    entries_with_eid = sum(1 for entries in lookup.values() for e in entries if e[3] and e[3].strip())
    total_entries = sum(len(entries) for entries in lookup.values())
    print(f"[DIAG] Lookup entries with entity_id: {entries_with_eid}/{total_entries}")

    # Step 5: Apply to result_df - match by call_id, then determine side by phone comparison
    matches_found = 0
    phone_mismatches = 0

    for idx in result_df.index:
        row = result_df.loc[idx]
        call_id = str(row.get(Config.COL_ORIGINAL_ID, '')).strip()

        # Get all entries for this call_id
        entries = lookup.get(call_id, [])
        if not entries:
            continue

        # Get phone numbers for both sides
        num_a = normalize_phone_for_output(row.get(Config.COL_NUM_A_OUT, ''))
        num_b = normalize_phone_for_output(row.get(Config.COL_NUM_B_OUT, ''))

        # Check each entry and determine which side it belongs to
        entry_matched = False
        for (phone, name, nickname, entity_id, status, id_number) in entries:
            if phone_numbers_match(phone, num_a):
                result_df.at[idx, Config.COL_VERIFIED_NAME_A] = name
                result_df.at[idx, Config.COL_VERIFIED_NICKNAME_A] = nickname
                result_df.at[idx, Config.COL_VERIFIED_ENTITY_ID_A] = entity_id
                result_df.at[idx, Config.COL_VERIFIED_STATUS_A] = status
                result_df.at[idx, Config.COL_VERIFIED_ID_NUMBER_A] = id_number
                matches_found += 1
                entry_matched = True
            elif phone_numbers_match(phone, num_b):
                result_df.at[idx, Config.COL_VERIFIED_NAME_B] = name
                result_df.at[idx, Config.COL_VERIFIED_NICKNAME_B] = nickname
                result_df.at[idx, Config.COL_VERIFIED_ENTITY_ID_B] = entity_id
                result_df.at[idx, Config.COL_VERIFIED_STATUS_B] = status
                result_df.at[idx, Config.COL_VERIFIED_ID_NUMBER_B] = id_number
                matches_found += 1
                entry_matched = True

        if not entry_matched and entries:
            phone_mismatches += 1
            if phone_mismatches <= 3:  # Only print first 3
                print(f"[DEBUG] Phone mismatch: call_id='{call_id}', num_a='{num_a}', num_b='{num_b}', lookup_phones={[e[0] for e in entries]}")

    print(f"[DEBUG] Final: {matches_found} verified names applied, {phone_mismatches} phone mismatches")

    return result_df


# ============================================================================
# SANITIZATION
# ============================================================================

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame by handling NaN and infinite values"""
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


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(cubes):
    """
    Main entry point for Cube Framework.
    v2.9.0: Extended Role Keywords & Removers
    
    Inputs:
    - cube1: Call data with content
    - cube5: Config with main_number
    - cube3: Enhancement table (id, title, nicknames, status, id_number) - OPTIONAL
    - cube4: Script 1 output (call_id, phone, original) - OPTIONAL
    """
    is_valid, error_msg = validate_input(cubes)
    if not is_valid:
        return pd.DataFrame({'error': [error_msg], 'status': ['validation_failed']})
    
    df = cubes[Config.INPUT_CUBE].copy()
    df2 = cubes[Config.INPUT_CUBE_2].copy()
    
    if not df2.empty and Config.COL_MAIN_NUMBER in df2.columns:
        main_number = df2[Config.COL_MAIN_NUMBER].iloc[0]
    else:
        main_number = None
    
    df_dedup, dedup_stats = deduplicate_by_latest_call(df)
    result_df = process_dataframe(df_dedup, main_number)
    
    # NEW v2.8.0: Integrate verified entities from cube3 + cube4
    cube3 = cubes.get(Config.INPUT_CUBE_3)
    cube4 = cubes.get(Config.INPUT_CUBE_4)
    
    # DIAGNOSTIC: Check cube availability
    print(f"[DIAG] Available cubes: {list(cubes.keys())}")
    print(f"[DIAG] Looking for cube3='{Config.INPUT_CUBE_3}', cube4='{Config.INPUT_CUBE_4}'")
    print(f"[DIAG] cube3 present: {cube3 is not None}, cube4 present: {cube4 is not None}")
    
    if cube3 is not None:
        print(f"[DIAG] cube3 type: {type(cube3).__name__}, shape: {cube3.shape if hasattr(cube3, 'shape') else 'N/A'}")
        if isinstance(cube3, pd.DataFrame) and not cube3.empty:
            print(f"[DIAG] cube3 columns: {list(cube3.columns)}")
            # Check if 'id' column exists
            id_col = Config.CUBE3_COL_ID
            if id_col in cube3.columns:
                non_null = cube3[id_col].notna().sum()
                print(f"[DIAG] cube3 '{id_col}' column: {non_null}/{len(cube3)} non-null")
                if non_null > 0:
                    sample = cube3[cube3[id_col].notna()][id_col].head(3).tolist()
                    print(f"[DIAG] Sample '{id_col}' values: {sample}")
            else:
                print(f"[DIAG] WARNING: '{id_col}' column NOT FOUND in cube3!")
    
    if cube4 is not None:
        print(f"[DIAG] cube4 type: {type(cube4).__name__}, shape: {cube4.shape if hasattr(cube4, 'shape') else 'N/A'}")

    if cube3 is not None and cube4 is not None:
        if isinstance(cube3, pd.DataFrame) and isinstance(cube4, pd.DataFrame):
            if not cube3.empty and not cube4.empty:
                result_df = integrate_verified_entities(result_df, cube3, cube4)
                print(f"v2.9.0: Verified entity integration complete")
            else:
                print(f"[DIAG] WARNING: cube3 empty={cube3.empty}, cube4 empty={cube4.empty}")
    else:
        print(f"[DIAG] WARNING: cube3 or cube4 not provided - skipping entity integration")
    
    result_df = sanitize_dataframe(result_df)
    
    print(f"v2.9.0 Extraction complete: {len(result_df)} records processed")
    print(f"Deduplication: {dedup_stats}")
    print(f"Output columns ({len(result_df.columns)}): {list(result_df.columns)}")
    
    # DIAGNOSTIC: Final entity_id check
    eid_a_col = Config.COL_VERIFIED_ENTITY_ID_A
    eid_b_col = Config.COL_VERIFIED_ENTITY_ID_B
    
    print(f"[FINAL] Checking columns: {eid_a_col}, {eid_b_col}")
    
    if eid_a_col in result_df.columns:
        non_empty_a = result_df[eid_a_col].apply(
            lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None')
        ).sum()
        print(f"[FINAL] {eid_a_col} non-empty: {non_empty_a}/{len(result_df)}")
        if non_empty_a > 0:
            sample = result_df[result_df[eid_a_col].apply(
                lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None')
            )][eid_a_col].head(5).tolist()
            print(f"[FINAL] Sample {eid_a_col} values: {sample}")
    else:
        print(f"[FINAL] WARNING: {eid_a_col} column not in output!")
        
    if eid_b_col in result_df.columns:
        non_empty_b = result_df[eid_b_col].apply(
            lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None')
        ).sum()
        print(f"[FINAL] {eid_b_col} non-empty: {non_empty_b}/{len(result_df)}")
        if non_empty_b > 0:
            sample = result_df[result_df[eid_b_col].apply(
                lambda x: bool(x) and str(x).strip() not in ('', 'nan', 'None')
            )][eid_b_col].head(5).tolist()
            print(f"[FINAL] Sample {eid_b_col} values: {sample}")
    else:
        print(f"[FINAL] WARNING: {eid_b_col} column not in output!")
    
    return result_df


# ============================================================================
# EXECUTION
# ============================================================================

if 'cubes' in globals():
    result = main(cubes)


