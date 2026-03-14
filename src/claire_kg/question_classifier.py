"""
Schema selection: rule-based question analysis without LLM.

Drives which schema and examples CypherGenerator uses by extracting from the
natural language question: primary datasets (CVE, CWE, CAPEC, ATT&CK, NICE,
DCWF, etc.), crosswalk groups, complexity, intent types, expected schema pack,
key properties, and potential failure patterns (C/D/E/F). Used by
LLMOrchestrator and QueryOrchestrator to pass classification_metadata into
generate_cypher; also provides is_heavy_question and is_out_of_domain for
timeout/limit and guardrails.

Module layout: enums (Dataset, IntentType, FailurePattern) → helpers
(is_heavy_question, is_out_of_domain) → ClassificationResult → QuestionClassifier.
Entry point: QuestionClassifier.classify(question) → ClassificationResult;
optional should_use_rag(question, classification_result) for RAG vs Cypher.
"""

import re
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum

# -----------------------------------------------------------------------------
# Enums: Dataset, IntentType, FailurePattern
# -----------------------------------------------------------------------------


class Dataset(str, Enum):
    """Supported datasets."""

    CVE = "CVE"
    CWE = "CWE"
    CAPEC = "CAPEC"
    ATTACK = "ATT&CK"
    NICE = "NICE"
    DCWF = "DCWF"
    ASSET = "Asset"
    MITIGATION = "Mitigation"
    CATEGORY = "Category"  # CAPEC category classification


class IntentType(str, Enum):
    """Intent type taxonomy."""

    # Core operations
    LOOKUP = "lookup"
    CVE_LOOKUP = "cve_lookup"  # Specific CVE lookup (CVSS, description)
    COUNT = "count"
    LIST = "list"
    TRAVERSE = "traverse"
    FILTER = "filter"

    # Semantic & text-based
    SEMANTIC_SEARCH = "semantic_search"
    SIMILARITY_SEARCH = "similarity_search"  # RAG: vector similarity search
    CATEGORIZE = "categorize"
    RELATED = "related"

    # Relationship & path
    TRAVERSE_2HOP = "traverse_2hop"
    MULTI_HOP = "multi_hop"
    PATH_FIND = "path_find"
    INFER = "infer"

    # Boolean & set
    BOOLEAN_AND = "boolean_and"
    BOOLEAN_OR = "boolean_or"
    BOOLEAN_NOT = "boolean_not"
    SET_INTERSECTION = "set_intersection"
    SET_UNION = "set_union"

    # Aggregation & analytics
    AGGREGATE = "aggregate"
    TEMPORAL = "temporal"
    RECENT = "recent"
    STATISTICAL = "statistical"
    COVERAGE = "coverage"

    # Domain-specific
    MITIGATION = "mitigation"
    WORKFORCE = "workforce"
    VENDOR = "vendor"
    PLATFORM = "platform"
    TACTIC = "tactic"

    # Complex multi-domain
    CROSS_FRAMEWORK = "cross_framework"
    COMPLETE_CHAIN = "complete_chain"
    DEFENSE_STRATEGY = "defense_strategy"
    ATTACK_SURFACE = "attack_surface"

    # Special
    CONTEXTUAL = "contextual"
    VOCABULARY_MAPPING = "vocabulary_mapping"
    AMBIGUOUS = "ambiguous"
    COMPOSITE = "composite"


class FailurePattern(str, Enum):
    """Known failure patterns (count, ordering, statistical, etc.) for evaluator hints."""

    PATTERN_C = "pattern_c"
    PATTERN_D = "pattern_d"
    PATTERN_E = "pattern_e"
    PATTERN_F = "pattern_f"


# -----------------------------------------------------------------------------
# Helpers: is_heavy_question, is_out_of_domain
# -----------------------------------------------------------------------------


def is_heavy_question(question: str) -> bool:
    """Detect questions that need longer timeout and/or smaller result limit.

    Heavy questions include:
    - Similarity/embedding searches (slow vector comparisons)
    - Large result set queries (all work roles, list every X)
    - Multi-hop traversals (CVE → CWE → CAPEC → ATT&CK)
    - Cross-framework comparisons (NICE vs DCWF overlap)

    Usage:
        from claire_kg.question_classifier import is_heavy_question

        if is_heavy_question(question):
            timeout = 300  # 5 minutes
            limit = 15     # Smaller result set
        else:
            timeout = 180  # 3 minutes
            limit = 25     # Normal result set

    Args:
        question: The user's question text

    Returns:
        True if question is expected to be slow/resource-intensive
    """
    if not question:
        return False

    ql = question.lower()

    # Heavy question patterns
    heavy_patterns = [
        # Similarity search (requires embedding comparisons)
        r"similar\s+to",
        r"like\s+cve-",
        r"vulnerabilities?\s+like",
        r"similar\s+vulnerabilities",
        r"related\s+vulnerabilities",
        # Large result sets
        r"all\s+work\s*roles?",
        r"list\s+(all|every)",
        r"show\s+(all|every)",
        r"every\s+\w+\s+in",
        # Cross-framework comparisons
        r"overlap\s+between",
        r"least\s+overlap",
        r"unique\s+to\s+only\s+one",
        r"nice\s+and\s+dcwf",
        r"dcwf\s+and\s+nice",
        r"both\s+frameworks?",
        # Multi-hop traversals (3+ entity types)
        r"vulnerabilities.*weaknesses.*attack\s*patterns",
        r"cve.*cwe.*capec.*att&ck",
        r"attack\s+chain",
        r"complete\s+chain",
        # Broad semantic searches
        r"everything\s+(about|related)",
        r"all\s+information\s+(about|on)",
    ]

    return any(re.search(pattern, ql) for pattern in heavy_patterns)


def is_out_of_domain(question: str) -> bool:
    """Detect questions that are outside the cybersecurity/KG domain.

    Out-of-domain questions include:
    - General knowledge (history, science, geography, sports, etc.)
    - Personal questions (opinions, feelings, recommendations)
    - Off-topic questions not related to cybersecurity

    In-domain questions (should return False):
    - CVE, CWE, CAPEC, ATT&CK, MITRE related
    - Vulnerabilities, weaknesses, attack patterns
    - NICE, DCWF work roles
    - Mitigations, defenses, security controls
    - Any cybersecurity-related terminology

    Usage:
        from claire_kg.question_classifier import is_out_of_domain

        if is_out_of_domain(question):
            return "This question is outside the scope of the cybersecurity knowledge graph."

    Args:
        question: The user's question text

    Returns:
        True if question is clearly outside the cybersecurity/KG domain
    """
    if not question:
        return False

    ql = question.lower().strip()

    # First check: Does the question contain ANY cybersecurity-related terms?
    # If so, it's likely in-domain even if phrased unusually
    cybersecurity_terms = [
        # Entity types
        r"\bcve\b",
        r"\bcwe\b",
        r"\bcapec\b",
        r"att&ck",
        r"attack",
        r"mitre",
        r"\bnice\b",
        r"\bdcwf\b",
        r"workforce",
        # Core concepts
        r"vulnerabilit",
        r"weakness",
        r"exploit",
        r"threat",
        r"risk",
        r"malware",
        r"ransomware",
        r"phishing",
        r"injection",
        r"overflow",
        r"authentication",
        r"authorization",
        r"encryption",
        r"decrypt",
        r"firewall",
        r"intrusion",
        r"breach",
        r"compromise",
        r"patch",
        # Security domains
        r"cyber",
        r"secur",
        r"hack",
        r"penetrat",
        r"forensic",
        r"incident",
        r"response",
        r"defense",
        r"mitigation",
        r"remediat",
        r"compliance",
        r"audit",
        r"access\s+control",
        r"privilege",
        # Technical terms
        r"network",
        r"protocol",
        r"port",
        r"server",
        r"endpoint",
        r"malicious",
        r"payload",
        r"backdoor",
        r"trojan",
        r"worm",
        r"denial\s+of\s+service",
        r"dos",
        r"ddos",
        r"botnet",
        r"cross[\-\s]?site",
        r"xss",
        r"csrf",
        r"sqli",
        r"rce",
        r"buffer",
        r"heap",
        r"stack",
        r"memory",
        r"kernel",
        # Frameworks and standards
        r"nist",
        r"iso\s*27",
        r"pci",
        r"hipaa",
        r"gdpr",
        r"sox",
        r"owasp",
        r"sans",
        r"cis\s+benchmark",
        r"stix",
        r"taxii",
    ]

    # If any cybersecurity term is found, it's in-domain
    for term in cybersecurity_terms:
        if re.search(term, ql):
            return False  # In-domain

    # Second check: Does the question match common out-of-domain patterns?
    out_of_domain_patterns = [
        # General knowledge questions
        r"(who|what)\s+(is|was|are|were)\s+the\s+(greatest|best|most\s+famous|first)",
        r"(who|what)\s+(invented|discovered|created|founded)",
        r"(when|where)\s+(did|was|were)\s+\w+\s+(born|die|happen|occur)",
        r"capital\s+of\s+\w+",
        r"population\s+of",
        r"(how|what)\s+(tall|old|big|long|far|much\s+does)",
        # Science/history (non-cyber)
        r"(scientist|physicist|chemist|biologist|mathematician)",
        r"(planet|star|galaxy|universe|solar\s+system)",
        r"(dinosaur|fossil|evolution|species)",
        r"(world\s+war|revolution|empire|dynasty|ancient)",
        r"(recipe|ingredient|cook|bake)",
        r"(weather|climate|temperature|rain|snow)",
        # Entertainment/sports
        r"(movie|film|actor|actress|director|oscar)",
        r"(song|music|album|band|singer|concert)",
        r"(sport|team|player|championship|olympic|world\s+cup)",
        r"(game|video\s+game|playstation|xbox|nintendo)",
        # Personal/opinion
        r"(your\s+favorite|do\s+you\s+like|what\s+do\s+you\s+think)",
        r"(recommend|suggest|should\s+i)",
        r"(how\s+are\s+you|how\s+do\s+you\s+feel)",
        r"(tell\s+me\s+a\s+joke|tell\s+me\s+a\s+story)",
        # Conversational/meta
        r"(hello|hi|hey|good\s+(morning|afternoon|evening))",
        r"(thank\s+you|thanks|goodbye|bye)",
        r"(who\s+are\s+you|what\s+are\s+you|are\s+you\s+an?\s+ai)",
    ]

    for pattern in out_of_domain_patterns:
        if re.search(pattern, ql):
            return True  # Out-of-domain

    # Third check: Very short questions without any technical substance
    # "Who's the greatest scientist?" - short, no cyber terms
    words = ql.split()
    if len(words) <= 6:
        # Short question with no cyber terms detected above = likely out-of-domain
        # But be conservative - only flag obvious cases
        obvious_off_topic = [
            r"^who'?s?\s+the\s+(greatest|best|most)",
            r"^what'?s?\s+(the\s+)?(capital|population|weather)",
            r"^(how|when|where)\s+(did|was|is)\s+\w+\s+(born|die|invented)",
        ]
        for pattern in obvious_off_topic:
            if re.search(pattern, ql):
                return True

    # Default: assume in-domain (conservative approach)
    # Better to attempt a KG query than wrongly refuse a valid question
    return False


# -----------------------------------------------------------------------------
# ClassificationResult and QuestionClassifier
# -----------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Schema-selection result: datasets, crosswalks, complexity, intents, schema pack, failure pattern."""

    primary_datasets: List[str]
    crosswalk_groups: List[str]
    complexity_level: str  # easy, medium, hard
    intent_types: List[str]
    expected_schema_pack: List[str]
    key_properties: List[str]
    potential_failure_pattern: Optional[str] = None


class QuestionClassifier:
    """Rule-based schema selector: no LLM; patterns and metadata only."""

    def __init__(self, use_metadata: bool = True):
        """Initialize with optional cached NICE/DCWF role names from dataset_metadata.

        Args:
            use_metadata: If True, load WORKROLE_METADATA role names for exact matching.
                         If False, use pattern-based detection only (no metadata).
        """
        self.use_metadata = use_metadata

        if use_metadata:
            # Import cached role names from dataset_metadata to avoid database lookups
            try:
                from .dataset_metadata import WORKROLE_METADATA

                self.KNOWN_DCWF_ROLES = WORKROLE_METADATA.get("role_names", {}).get(
                    "DCWF", []
                )
                self.KNOWN_NICE_ROLES = WORKROLE_METADATA.get("role_names", {}).get(
                    "NICE", []
                )
            except ImportError:
                # Fallback if metadata not available
                self.KNOWN_DCWF_ROLES = []
                self.KNOWN_NICE_ROLES = []
        else:
            # Pattern-based mode: don't load metadata
            self.KNOWN_DCWF_ROLES = []
            self.KNOWN_NICE_ROLES = []

    # Fallback role list if dataset_metadata.WORKROLE_METADATA not available
    KNOWN_DCWF_ROLES_FALLBACK = [
        "ai adoption specialist",
        "ai innovation leader",
        "ai risk and ethics specialist",
        "ai test & evaluation specialist",
        "ai/ml specialist",
        "access network operator",
        "all-source analyst",
        "all-source collection manager",
        "all-source collection requirements manager",
        "authorizing official/designated representative",
        "comsec manager",
        "control systems security specialist",
        "cyber crime investigator",
        "cyber defense analyst",
        "cyber defense forensics analyst",
        "cyber defense incident responder",
        "cyber defense infrastructure support specialist",
        "cyber instructional curriculum developer",
        "cyber instructor",
        "cyber intelligence planner",
        "cyber legal advisor",
        "cyber operations planner",
        "cyber policy and strategy planner",
        "cyber workforce developer and manager",
        "cyberspace capability developer",
        "cyberspace operator",
        "data analyst",
        "data architect",
        "data officer",
        "data operations specialist",
        "data scientist",
        "data steward",
        "database administrator",
        "devsecops specialist",
        "digital network exploitation analyst",
        "enterprise architect",
        "executive cyber leader",
        "exploitation analyst",
        "forensics analyst",
        "host analyst",
        "it investment/portfolio manager",
        "it program auditor",
        "it project manager",
        "information systems security developer",
        "information systems security manager",
        "joint targeting analyst",
        "knowledge manager",
        "multi-disciplined language analyst",
        "network analyst",
        "network operations specialist",
        "network technician",
        "privacy compliance manager",
        "product designer user interface (ui)",
        "product manager",
        "product support manager",
        "program manager",
        "red team specialist",
        "research & development specialist",
        "secure software assessor",
        "security architect",
        "security control assessor",
        "service designer user experience (ux)",
        "software developer",
        "software test & evaluation specialist",
        "software/cloud architect",
        "system administrator",
        "system testing and evaluation specialist",
        "systems developer",
        "systems requirements planner",
        "systems security analyst",
        "target analyst reporter",
        "target digital network analyst",
        "technical support specialist",
        "vulnerability assessment analyst",
    ]

    # Dataset detection patterns
    DATASET_PATTERNS = {
        Dataset.CVE: [
            r"cve-\d{4}-\d+",
            r"\bcves?\b",  # Match both "CVE" and "CVEs"
            r"\bvulnerabilit",
            r"\bcvss",
            r"\bseverity\b",
            r"\bcritical\b",  # Severity level: Critical
            r"\bhigh\b",  # Severity level: High (context-dependent, but common for CVEs)
            r"\bmedium\b",  # Severity level: Medium
            r"\blow\b",  # Severity level: Low
        ],
        Dataset.CWE: [
            r"cwe-\d+",
            r"\bcwes?\b",  # Match both "CWE" and "CWEs" (plural)
            r"\bweakness",
            r"\bweaknesses\b",
        ],
        Dataset.CAPEC: [
            r"capec-\d+",
            r"\bcapec\b",
            r"attack pattern",
            r"attack patterns",
        ],
        Dataset.CATEGORY: [
            # Only detect Category in CAPEC context (CAPEC categories, not generic categorization)
            r"capec.*category|attack pattern.*category",
            r"labeled as.*category",  # "labeled as Software category" is CAPEC-specific
            r"category.*capec",
        ],
        Dataset.ATTACK: [
            r"\bt\d{4}\b",
            r"\bta\d{4}\b",
            r"\btechnique",
            r"\btactics?\b",
            r"\battack\b",
            r"\bmitre\b",
        ],
        Dataset.NICE: [
            r"\bnice\b",
            r"work role(?!\s*\d+)",  # "work role" but NOT "work role 441" (that's DCWF)
            r"work roles(?!\s*\d+)",  # "work roles" but NOT "work roles 441"
            r"\btask",
            r"\btasks\b",
            r"\bskill",
            r"\bskills\b",
            r"\bknowledge\b",
            r"cyber defense",
            # Common job titles from schema packs (WorkRole nodes)
            r"\badministrator\b",  # System Administrator, Network Administrator
            r"\banalyst\b",  # Cyber Defense Analyst, Security Analyst
            r"\bengineer\b",  # Security Engineer, Network Engineer
            r"\bmanager\b",  # Security Manager, IT Manager
            r"\bcoordinator\b",  # Security Coordinator
            r"\bspecialist\b",  # Security Specialist
            r"\bforensics analyst\b",  # From schema packs
            r"\bvulnerability assessment\b",  # Vulnerability Assessment Analyst
            r"\bincident response\b",  # Incident Response roles
            r"\bpenetration test\b",  # Penetration Tester
            r"\bthreat hunt\b",  # Threat Hunter
        ],
        Dataset.DCWF: [
            r"\bdcwf\b",
            r"specialty area",  # Note: Also matches NICE SpecialtyArea - use context to disambiguate
            r"forensics analyst",
            r"work role \d+",
        ],
        Dataset.ASSET: [
            r"\basset",
            r"\bassets\b",
            r"\bcpe\b",
            r"\bcpes\b",
            r"\bvendor\b",
            r"\bproduct\b",
        ],
        Dataset.MITIGATION: [
            r"\bmitigation",
            r"\bmitigations\b",
        ],
    }

    # Crosswalk detection patterns
    CROSSWALK_PATTERNS = {
        "CVE<->CWE": [r"cve.*cwe|cwe.*cve", r"vulnerabilit.*weakness"],
        "CVE<->Asset": [r"cve.*asset|cve.*cpe|vulnerabilit.*affect"],
        "CVE<->ATT&CK": [r"cve.*technique|cve.*attack|vulnerabilit.*exploit"],
        "CAPEC<->ATT&CK": [r"capec.*technique|attack pattern.*technique"],
        "CAPEC<->CWE": [r"capec.*weakness|attack pattern.*weakness"],
        "NICE<->ATT&CK": [r"work role.*technique|nice.*attack"],
        "NICE<->DCWF": [r"nice.*dcwf|dcwf.*nice"],
        "CWE<->Mitigation": [r"cwe.*mitigation|weakness.*mitigation"],
    }

    # Intent detection patterns
    INTENT_PATTERNS = {
        IntentType.LOOKUP: [r"what is", r"what does", r"describe", r"show.*pattern"],
        # CVE lookup: specific CVE + CVSS/description request
        IntentType.CVE_LOOKUP: [
            r"cvss.*score.*cve-\d{4}-\d+",
            r"cve-\d{4}-\d+.*cvss",
            r"cve-\d{4}-\d+.*description",
            r"description.*cve-\d{4}-\d+",
            r"what is.*cve-\d{4}-\d+",
            r"what is the.*cvss.*score",
        ],
        # COUNT: drives Pattern C (count queries), prompt examples, and count-first answers
        IntentType.COUNT: [
            r"how many",
            r"count",
            r"number of",
            r"total number",
            r"total count",
        ],
        IntentType.LIST: [r"list all", r"show me", r"which.*are", r"what.*are"],
        IntentType.TRAVERSE: [
            r"linked to",
            r"related to",
            r"connected to",
            r"belongs? to",
        ],
        IntentType.FILTER: [r"find.*with", r"above", r"below", r">", r"<"],
        IntentType.SEMANTIC_SEARCH: [
            r"show me.*for",
            r"buffer overflow",
            r"sql injection",
        ],
        IntentType.SIMILARITY_SEARCH: [
            r"similar to",
            r"most similar",
            r"most like",
            r"like.*cve-",
            r"like.*cwe-",
            r"like.*capec-",
            r"like.*t\d{4}",
            r"find.*similar",
            r"show.*similar",
            r"group.*by similarity",
            r"cluster.*by",
            r"related.*conceptually",
            r"conceptually similar",
        ],
        IntentType.CATEGORIZE: [r"categorized", r"labeled as", r"category"],
        IntentType.RECENT: [r"most recent", r"latest", r"newest"],
        # TEMPORAL: Require temporal context, not just any 4-digit number
        # This prevents false positives from years in identifiers (e.g., CVE-2024-20439)
        IntentType.TEMPORAL: [
            r"\d{4}.*(published|discovered|reported|released|created|announced)",  # Year + temporal verb
            r"(published|discovered|reported|released|created|announced).*\d{4}",  # Temporal verb + year
            r"(in|during|since|before|after)\s+\d{4}",  # Temporal preposition + year
            r"year\s+\d{4}",  # Explicit "year 2024"
            r"\d{4}.*year",  # "2024 year"
            r"date",  # Explicit date mention
        ],
        IntentType.AGGREGATE: [
            r"top \d+",
            r"most common",
            r"most prevalent",
            r"most critical",
        ],
        IntentType.BOOLEAN_AND: [r"both.*and", r"address both"],
        IntentType.BOOLEAN_OR: [r"\bor\b", r"either.*or"],
        IntentType.BOOLEAN_NOT: [r"no.*linked", r"have no", r"not.*mitigation"],
        IntentType.MITIGATION: [r"mitigation", r"mitigations"],
        IntentType.WORKFORCE: [
            r"work role",
            r"task",
            r"skill",
            r"knowledge",
            r"workforce",
            r"what does.*do",  # "What does X do?" → workforce/work role query
            r"what is.*responsible",  # "What is X responsible for?"
            r"what.*required.*role",  # "What is required for role X?"
        ],
        IntentType.VENDOR: [
            r"vendor",
            r"product",
            r"microsoft",
            r"linux",
            r"affects?.*product",
        ],
        IntentType.PLATFORM: [r"platform", r"windows", r"target.*platform"],
        IntentType.TACTIC: [r"tactic", r"tactics", r"under.*tactic"],
        IntentType.MULTI_HOP: [r"inferred through", r"via.*and", r"through.*and"],
        IntentType.PATH_FIND: [r"path from", r"attack path", r"attack chain"],
        IntentType.COMPLETE_CHAIN: [r"complete.*chain", r"from.*to.*to"],
        IntentType.CROSS_FRAMEWORK: [
            r"across.*framework",
            r"all framework",
            r"all.*datasets",
        ],
        IntentType.STATISTICAL: [r"correlation", r"relationship between"],
        IntentType.COVERAGE: [r"coverage", r"comprehensive"],
        IntentType.DEFENSE_STRATEGY: [r"defense.*strategy", r"defense.*depth"],
        IntentType.ATTACK_SURFACE: [r"attack surface", r"threat landscape"],
        IntentType.CONTEXTUAL: [r"based on.*previous", r"previous results"],
        IntentType.VOCABULARY_MAPPING: [
            r"buffer overflow",
            r"stack overflow",
            r"heap overflow",
        ],
        IntentType.INFER: [r"can be inferred", r"inferred through"],
    }

    # Failure pattern detection
    FAILURE_PATTERN_PATTERNS = {
        FailurePattern.PATTERN_C: [r"how many", r"count"],
        FailurePattern.PATTERN_D: [r"most recent", r"latest", r"newest", r"oldest"],
        FailurePattern.PATTERN_E: [
            r"skill.*defend.*cve",
            r"cve.*skill",
            r"top.*cve.*skill",
            r"workforce.*vulnerabilit",
            r"vulnerabilit.*workforce",
            r"work.*role.*cve",
            r"cve.*work.*role",
            r"nice.*cve",
            r"dcwf.*cve",
            r"technique.*workforce",
            r"workforce.*technique",
        ],
        FailurePattern.PATTERN_F: [
            r"correlation",
            r"relationship between",
            r"severity.*expertise",
        ],
    }

    # Complexity heuristics
    COMPLEXITY_INDICATORS = {
        "easy": [
            r"what is.*cve-\d+",
            r"what does.*cwe-\d+",
            r"show me.*pattern",
        ],
        "medium": [
            r"linked to.*via",
            r"connect.*to",
            r"\bcrosswalk\b(?!\s+relationships?)",  # Simple crosswalk mention (but NOT "crosswalk relationships")
        ],
        "hard": [
            r"inferred\s+through",
            r"complete.*chain",
            r"across.*all.*framework",
            r"across\s+all\s+cybersecurity\s+frameworks",  # Q101: more specific
            r"all\s+cybersecurity\s+frameworks",  # Q101
            r"crosswalk\s+relationships?.*(critical|effective|all)",  # Q101: crosswalk + modifiers
            r"correlation",
            r"attack\s+surface",
            r"complete\s+attack\s+surface",  # Q88
            r"complete\s+threat\s+landscape",  # Q93
        ],
    }

    def classify(self, question: str) -> ClassificationResult:
        """
        Run schema selection: datasets → crosswalks → complexity → intents → schema pack → key properties → failure pattern.

        Args:
            question: Natural language question

        Returns:
            ClassificationResult with all metadata for CypherGenerator and orchestrators
        """
        question_lower = question.lower()

        primary_datasets = self._detect_datasets(question_lower)
        crosswalk_groups = self._detect_crosswalks(question_lower, primary_datasets)
        complexity_level = self._detect_complexity(
            question_lower, primary_datasets, crosswalk_groups
        )
        intent_types = self._detect_intents(question_lower)
        expected_schema_pack = self._determine_schema_packs(
            primary_datasets, crosswalk_groups
        )
        key_properties = self._extract_key_properties(
            question_lower, primary_datasets, intent_types
        )
        potential_failure_pattern = self._detect_failure_patterns(
            question_lower, intent_types
        )

        return ClassificationResult(
            primary_datasets=primary_datasets,
            crosswalk_groups=crosswalk_groups,
            complexity_level=complexity_level,
            intent_types=intent_types,
            expected_schema_pack=expected_schema_pack,
            key_properties=key_properties,
            potential_failure_pattern=potential_failure_pattern,
        )

    def _detect_datasets(self, question_lower: str) -> List[str]:
        """Detect primary datasets mentioned in question.

        Root cause fixes:
        1. Pattern specificity: More specific patterns checked first to avoid false positives
        2. Context-aware matching: Check context before applying generic patterns
        3. Semantic relationships: Recognize semantic patterns (e.g., "vulnerabilities for X" → Asset)
        """
        detected = set()

        # ROOT CAUSE 1: Check specific patterns BEFORE generic ones to avoid ambiguity
        # Check for explicit IDs first (most specific)
        if re.search(r"capec-\d+", question_lower, re.IGNORECASE):
            detected.add(Dataset.CAPEC.value)
        if re.search(r"cwe-\d+", question_lower, re.IGNORECASE):
            detected.add(Dataset.CWE.value)
        if re.search(r"cve-\d{4}-\d+", question_lower, re.IGNORECASE):
            detected.add(Dataset.CVE.value)

        # ROOT CAUSE 4: Context-aware ATT&CK ID detection
        # T\d{4} patterns can be ATT&CK techniques OR NICE tasks
        # If question mentions "task T\d{4}", it's NICE, not ATT&CK
        is_nice_task_id = bool(
            re.search(r"task\s+t\d{4}", question_lower, re.IGNORECASE)
        )
        if not is_nice_task_id and re.search(
            r"\bt\d{4}\b", question_lower, re.IGNORECASE
        ):
            # No "task" context, so assume ATT&CK technique
            detected.add(Dataset.ATTACK.value)

        # ROOT CAUSE 2: Check for specific phrases that override generic patterns
        # "attack pattern" is CAPEC-specific, not ATT&CK
        has_attack_pattern = re.search(r"attack pattern", question_lower, re.IGNORECASE)
        if has_attack_pattern:
            detected.add(Dataset.CAPEC.value)

        # ROOT CAUSE 3: Context-aware vulnerability detection
        # If "vulnerability" appears in work role context, DON'T detect CVE
        # Pattern: "vulnerability [adjective] [noun]" where noun is a work role term
        work_role_context_patterns = [
            r"vulnerability\s+(assessment|analyst|engineer|specialist|manager)",
            r"vulnerability\s+assessment",
        ]
        is_work_role_context = any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in work_role_context_patterns
        )

        # Now process all patterns
        for dataset, patterns in self.DATASET_PATTERNS.items():
            # Skip CVE if it's a work role context
            if dataset == Dataset.CVE and is_work_role_context:
                continue

            # Skip ATT&CK if we already detected CAPEC from "attack pattern"
            # BUT ONLY if there's NO explicit ATT&CK indicator
            if dataset == Dataset.ATTACK:
                # ROOT CAUSE 4: Skip if this is actually a NICE Task ID (e.g., "task T0037")
                if is_nice_task_id:
                    continue
                if Dataset.CAPEC.value in detected and re.search(
                    r"attack pattern", question_lower, re.IGNORECASE
                ):
                    # Check for explicit ATT&CK indicators FIRST (before excluding)
                    has_explicit_attack = re.search(
                        r"\b(technique|techniques|tactic|tactics|t\d{4}|mitre)\b",
                        question_lower,
                        re.IGNORECASE,
                    )
                    if not has_explicit_attack:
                        continue  # No explicit ATT&CK indicator, so exclude it
                    # If explicit indicator exists, allow ATT&CK to be added

            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    detected.add(dataset.value)
                    break

        # Check for explicit DCWF indicators (DCWF is a separate dataset)
        # Must check DCWF BEFORE NICE to avoid false matches

        # First: Check if question contains a known DCWF role name (from cached metadata)
        # ONLY if metadata is enabled
        dcwf_role_detected = False
        if self.use_metadata:
            for role in self.KNOWN_DCWF_ROLES:
                if role in question_lower:
                    detected.add(Dataset.DCWF.value)
                    dcwf_role_detected = True
                    break

        # Also check for known NICE role names (if metadata enabled)
        nice_role_detected = False
        if self.use_metadata:
            for role in self.KNOWN_NICE_ROLES:
                if role in question_lower:
                    detected.add(Dataset.NICE.value)
                    nice_role_detected = True
                    break

        # ROOT CAUSE 4: Semantic disambiguation for roles that exist in both frameworks
        # If role detected in DCWF, check question context to determine primary dataset:
        # - Questions about "tasks" → NICE (DCWF doesn't have tasks)
        # - Questions about "abilities" → DCWF
        # - Explicit DCWF mentions (codes, "dcwf") → DCWF
        # - Explicit NICE mentions → NICE
        # BUT: Don't remove DCWF if there's a numeric work role code (those indicate DCWF via NICE)
        if dcwf_role_detected:
            # Check for semantic clues about which framework is primary
            has_nice_semantics = re.search(
                r"\b(task|tasks|knowledge|skill|skills)\b",
                question_lower,
                re.IGNORECASE,
            )
            has_dcwf_semantics = re.search(
                r"\b(ability|abilities|dcwf|work role \d+)\b",
                question_lower,
                re.IGNORECASE,
            )
            has_explicit_nice = re.search(r"\bnice\b", question_lower, re.IGNORECASE)
            has_explicit_dcwf = re.search(r"\bdcwf\b", question_lower, re.IGNORECASE)
            has_numeric_code = re.search(
                r"work role\s+\d+", question_lower, re.IGNORECASE
            )

            # If question has NICE semantics (tasks, knowledge, skills) BUT has numeric code, keep both (DCWF via NICE)
            if has_nice_semantics and has_numeric_code:
                # Numeric codes indicate DCWF, even if tasks are mentioned (Q33: "tasks...work role 441")
                # Keep both DCWF and NICE for "DCWF via NICE" queries
                if Dataset.DCWF.value not in detected:
                    detected.add(Dataset.DCWF.value)
                if Dataset.NICE.value not in detected:
                    detected.add(Dataset.NICE.value)
            elif (
                has_nice_semantics and not has_explicit_dcwf and not has_dcwf_semantics
            ):
                detected.discard(Dataset.DCWF.value)
                detected.add(Dataset.NICE.value)
            # If question has DCWF semantics or explicit DCWF mention, keep DCWF, don't add NICE unless explicit
            elif has_dcwf_semantics or has_explicit_dcwf:
                if not has_explicit_nice:
                    detected.discard(Dataset.NICE.value)

        # ROOT CAUSE 5: Specialty area disambiguation - MUST run BEFORE generic workforce logic
        # "specialty area" can match both DCWF and NICE (via DATASET_PATTERNS)
        # If question mentions "specialty area" but NOT "dcwf" explicitly, prefer NICE
        has_specialty_area = re.search(r"specialty area", question_lower, re.IGNORECASE)
        has_explicit_dcwf_mention = re.search(
            r"\bdcwf\b", question_lower, re.IGNORECASE
        )

        if (
            has_specialty_area
            and Dataset.DCWF.value in detected
            and Dataset.NICE.value in detected
        ):
            if not has_explicit_dcwf_mention:
                # No explicit DCWF mention, so this is likely a NICE SpecialtyArea question
                detected.discard(Dataset.DCWF.value)

        # Skills aligned with tasks are NICE-only (tasks don't exist in DCWF)
        # MUST run BEFORE generic workforce logic to prevent DCWF from being added back
        has_skills_align_task = re.search(
            r"skills?.*align.*task|task.*skills?.*align", question_lower, re.IGNORECASE
        )
        if has_skills_align_task:
            if Dataset.DCWF.value in detected:
                detected.remove(Dataset.DCWF.value)
            # Ensure NICE is detected if not already
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)

        # Second: Check for explicit DCWF keywords/codes
        # "DCWF framework" or numeric work role codes (e.g., "work role 441") indicate DCWF
        dcwf_indicators = [
            r"\bdcwf\b",
            r"work role\s+\d+",  # e.g., "work role 441" (DCWF codes are numeric)
            r"dcwf framework",
            r"dcwf.*work role",
            r"work role.*\d{3,}",  # DCWF codes are typically 3+ digits (e.g., 441)
        ]
        dcwf_keyword_detected = any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in dcwf_indicators
        )

        # Check if this is a "DCWF via NICE" scenario
        # "DCWF via NICE" means DCWF data is accessed through NICE nodes
        is_dcwf_via_nice = (
            "dcwf framework" in question_lower
            or re.search(
                r"work role\s+\d+|work role.*\d{3,}", question_lower, re.IGNORECASE
            )
            or (
                re.search(r"\bdcwf\b", question_lower, re.IGNORECASE)
                and (
                    "specialty area" in question_lower
                    or "work role" in question_lower
                    or "work roles" in question_lower
                )
            )
        )

        if dcwf_keyword_detected:
            detected.add(Dataset.DCWF.value)
            # For "DCWF (via NICE WorkRole)" queries: Keep NICE as well since DCWF is accessed via NICE nodes
            if is_dcwf_via_nice:
                # "DCWF framework", numeric codes, or DCWF with specialty area/work role = DCWF via NICE
                # Ensure NICE is also detected
                if Dataset.NICE.value not in detected:
                    detected.add(Dataset.NICE.value)
            elif r"\bnice\b" not in question_lower:
                # If DCWF explicitly mentioned but not "via NICE", remove NICE if it was auto-detected
                if Dataset.NICE.value in detected:
                    detected.remove(Dataset.NICE.value)

        # ROOT CAUSE 6: Generic workforce queries should detect BOTH DCWF and NICE
        # BUT: Tasks are NICE-specific (DCWF doesn't have tasks), so prefer NICE-only for task queries
        # Since they share WorkRole nodes (74/115 roles exist in both), detect both for generic queries
        # Patterns: "tasks", "work roles", workforce-related queries
        has_workforce_query = (
            re.search(r"\btasks?\b", question_lower, re.IGNORECASE)
            or re.search(r"work role", question_lower, re.IGNORECASE)
            or re.search(r"specialty area", question_lower, re.IGNORECASE)
        )

        # Tasks are NICE-specific: "tasks belong to", "tasks are associated with", etc.
        # If question is about tasks belonging to a role, prefer NICE only
        has_task_belongs_pattern = re.search(
            r"tasks?.*belongs?|belongs?.*tasks?", question_lower, re.IGNORECASE
        ) or re.search(
            r"tasks?.*associated|associated.*tasks?", question_lower, re.IGNORECASE
        )

        has_explicit_dcwf_only = (
            re.search(r"\bdcwf\b", question_lower, re.IGNORECASE)
            and not Dataset.NICE.value in detected
        )
        has_explicit_nice_only = (
            re.search(r"\bnice\b", question_lower, re.IGNORECASE)
            and not Dataset.DCWF.value in detected
        )

        # If it's a task-specific query, prefer NICE only (tasks don't exist in DCWF)
        # BUT: If there's a numeric work role code, keep both (DCWF via NICE for Q33)
        if has_task_belongs_pattern:
            has_numeric_code_in_question = re.search(
                r"work role\s+\d+", question_lower, re.IGNORECASE
            )
            if has_numeric_code_in_question:
                # Numeric codes indicate DCWF, even with "tasks associated" (Q33)
                # Add DCWF first, then ensure both are present
                detected.add(Dataset.DCWF.value)
                if Dataset.NICE.value not in detected:
                    detected.add(Dataset.NICE.value)
            else:
                # No numeric code, so this is NICE-only (tasks don't exist in DCWF)
                if Dataset.DCWF.value in detected:
                    detected.remove(Dataset.DCWF.value)
                # Ensure NICE is detected if not already
                if Dataset.NICE.value not in detected:
                    detected.add(Dataset.NICE.value)

        # If it's a generic workforce query (not task-specific) and no explicit framework restriction
        # BUT: Respect specialized restrictions (specialty area, skills align task, task belongs)
        # ALSO: Generic "show me X work roles" queries should prefer NICE-only unless explicitly comparing frameworks
        elif (
            has_workforce_query
            and not has_explicit_dcwf_only
            and not has_explicit_nice_only
        ):
            # Check if specialized restrictions were applied
            specialty_restricted_to_nice = (
                has_specialty_area
                and not has_explicit_dcwf_mention
                and Dataset.NICE.value in detected
                and Dataset.DCWF.value not in detected
            )
            skills_task_restricted = (
                has_skills_align_task
                and Dataset.NICE.value in detected
                and Dataset.DCWF.value not in detected
            )

            # Generic "show me/list X work roles" without explicit framework mention should prefer NICE-only
            # (These are typically NICE-specific queries unless explicitly about DCWF or both frameworks)
            is_generic_list_query = (
                re.search(
                    r"show me|list.*work role|work role.*show",
                    question_lower,
                    re.IGNORECASE,
                )
                and not has_explicit_dcwf_mention
                and not re.search(r"both|compare|each", question_lower, re.IGNORECASE)
            )

            # Generic task queries (like "forensics-related tasks", "show me X tasks") should detect both DCWF and NICE
            # since they share WorkRole nodes and tasks can belong to both frameworks (Q36-38)
            is_generic_task_query = (
                re.search(
                    r"show me.*tasks?|tasks?.*show", question_lower, re.IGNORECASE
                )
                and not has_explicit_dcwf_mention
                and not has_explicit_nice_only
                and not has_task_belongs_pattern  # Exclude task_belongs queries (those are role-specific)
            )

            # Only add both if not restricted by specialized checks AND not a generic list query
            # BUT: Generic task queries should include both (Q37)
            if is_generic_task_query:
                # Ensure both DCWF and NICE are detected for generic task queries
                if Dataset.NICE.value not in detected:
                    detected.add(Dataset.NICE.value)
                if Dataset.DCWF.value not in detected:
                    detected.add(Dataset.DCWF.value)
            elif not (
                specialty_restricted_to_nice
                or skills_task_restricted
                or is_generic_list_query
            ):
                if Dataset.NICE.value in detected or Dataset.DCWF.value in detected:
                    # If one is detected, add the other unless explicitly restricted
                    if Dataset.NICE.value in detected and not re.search(
                        r"\bnice\b.*only|\bonly.*nice\b", question_lower, re.IGNORECASE
                    ):
                        detected.add(Dataset.DCWF.value)
                    if Dataset.DCWF.value in detected and not re.search(
                        r"\bdcwf\b.*only|\bonly.*dcwf\b", question_lower, re.IGNORECASE
                    ):
                        detected.add(Dataset.NICE.value)

        # For generic "what does X do" workforce queries: detect BOTH DCWF and NICE
        # Since they share WorkRole nodes, both datasets should be checked
        # Many roles (74/115) are DCWF-aligned, so include DCWF even if NICE detected
        if re.search(r"what does.*do", question_lower, re.IGNORECASE):
            job_title_keywords = [
                r"\badministrator\b",
                r"\banalyst\b",
                r"\bengineer\b",
                r"\bmanager\b",
                r"\bcoordinator\b",
                r"\bspecialist\b",
                r"\bofficer\b",
                r"\bdirector\b",
                r"\bconsultant\b",
            ]
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in job_title_keywords
            ):
                # Generic workforce query - could be NICE or DCWF
                # Detect BOTH since they share WorkRole nodes (query generator can search both)
                detected.add(Dataset.DCWF.value)
                detected.add(Dataset.NICE.value)

        # ROOT CAUSE 3: Semantic relationship detection
        # Pattern: "vulnerabilities [preposition] [vendor/product]" → Asset dataset needed
        # This recognizes the semantic structure, not just specific vendor names
        asset_relationship_patterns = [
            r"vulnerabilit.*\b(for|affecting|that affect|affects?)\s+\w+",
            r"vulnerabilit.*\b(vendor|product)\b",
            r"affected\s+(systems?|assets?|products?|platforms?)",  # "affected systems", "affected assets"
            r"systems?\s+(affected|vulnerable|exploited)",  # "systems affected", "systems vulnerable"
        ]
        if Dataset.CVE.value in detected or "vulnerabilit" in question_lower:
            # Check for semantic patterns indicating Asset filtering
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in asset_relationship_patterns
            ):
                detected.add(Dataset.ASSET.value)
            # Also check if question mentions common vendor/product terms in context of "for/affecting"
            elif re.search(
                r"\b(for|affecting|affects?)\s+(linux|windows|microsoft|apple|oracle|apache|nginx|cisco|vmware|ios|android)\b",
                question_lower,
                re.IGNORECASE,
            ):
                detected.add(Dataset.ASSET.value)

        # ROOT CAUSE 6: Semantic attack pattern detection
        # When queries mention "X attacks" (web application, phishing, persistence, etc.)
        # AND mention ATT&CK techniques/tactics, they should also detect CAPEC
        # Pattern: "[attack type] attacks" + "techniques/tactics" → CAPEC + ATT&CK
        # Also: "phishing-related CAPEC attacks" + "tactics" → CAPEC already detected, ensure ATT&CK
        attack_type_patterns = [
            r"\b(web application|phishing|persistence|injection|sql injection|buffer overflow|xss|cross-site)\s+attacks?",
            r"attacks?\s+(related to|involving|associated with)\s+(web application|phishing|persistence)",
            r"(phishing|web application|persistence).*related.*\b(capec|attack pattern)",  # Q47: "phishing-related CAPEC"
        ]

        has_attack_type_mention = any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in attack_type_patterns
        )

        has_attack_technique_context = bool(
            re.search(
                r"\b(technique|techniques|tactic|tactics|related to|associated with|used by|work with)\b",
                question_lower,
                re.IGNORECASE,
            )
        )

        # If question mentions attack types AND ATT&CK context, add CAPEC
        if has_attack_type_mention and has_attack_technique_context:
            if Dataset.ATTACK.value in detected and Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)
            # Reverse: If CAPEC is mentioned with attack types and tactics, ensure ATT&CK is detected
            elif (
                Dataset.CAPEC.value in detected and Dataset.ATTACK.value not in detected
            ):
                if has_attack_technique_context:  # Mentions tactics/techniques
                    detected.add(Dataset.ATTACK.value)

        # ROOT CAUSE 14: CAPEC detection for exploit/attack pattern relationships
        # Pattern 1: "exploit [weakness/XSS/buffer overflow]" → CAPEC (attack patterns exploit weaknesses)
        # Q74: "techniques commonly used to exploit XSS weaknesses" → CWE + CAPEC + ATT&CK
        has_exploit_weakness = bool(
            re.search(
                r"\b(exploit|exploiting|exploited|exploits)\s+.*(weakness|weaknesses|vulnerability|vulnerabilities|xss|buffer overflow|sql injection|injection)",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_exploit_weakness:
            # If CWE is detected (weaknesses) or ATT&CK is detected (techniques), add CAPEC
            if (
                Dataset.CWE.value in detected or Dataset.ATTACK.value in detected
            ) and Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)

        # Pattern 2: "attack path from CVE to ATT&CK" → CVE→CWE→CAPEC→ATT&CK (Q73)
        # "attack chain from vulnerability" → CVE→CWE→CAPEC→ATT&CK (Q85)
        # BUT: If question asks for "attack chain linking Technique to CVEs and systems",
        # prefer direct relationships (CAN_BE_EXPLOITED_BY) over indirect paths
        has_attack_path_chain = bool(
            re.search(
                r"\b(attack\s+path|attack\s+chain|complete\s+attack|from\s+vulnerability.*to|from\s+cve.*to.*technique)",
                question_lower,
                re.IGNORECASE,
            )
        )
        # Check if query explicitly asks for "full" or "complete" attack chain
        # Full/complete chains should include CAPEC patterns (how techniques are executed)
        # even when using direct CVE→Technique relationships
        has_full_complete_chain = bool(
            re.search(
                r"\b(full|complete|entire|whole)\s+attack\s+chain",
                question_lower,
                re.IGNORECASE,
            )
        )
        # Check if query is asking for direct Technique→CVE relationship (not indirect via CAPEC)
        has_direct_technique_cve = bool(
            re.search(
                r"\b(technique|techniques?)\s+.*(to|linking|connected to|exploit).*(cve|cves|vulnerabilit|affected systems?)",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_attack_path_chain:
            # If both CVE and ATT&CK are detected, add CAPEC and CWE (full chain)
            # BUT: If query explicitly asks for direct Technique→CVE relationship, don't force indirect path
            # EXCEPTION: If question asks for "full" or "complete" attack chain, include CAPEC
            # because full chains need both direct relationships (which CVEs) AND attack patterns (how)
            if Dataset.CVE.value in detected and Dataset.ATTACK.value in detected:
                # Add CAPEC/CWE if:
                # 1. NOT asking for direct relationship (use indirect path), OR
                # 2. Asking for "full/complete" attack chain (need both direct relationships AND CAPEC patterns)
                if not has_direct_technique_cve or has_full_complete_chain:
                    if Dataset.CAPEC.value not in detected:
                        detected.add(Dataset.CAPEC.value)
                    # Only add CWE if not asking for direct relationship (CWE not needed for direct path)
                    if not has_direct_technique_cve:
                        if Dataset.CWE.value not in detected:
                            detected.add(Dataset.CWE.value)
                # Always add Asset if "affected systems" is mentioned
                if re.search(r"affected\s+systems?", question_lower, re.IGNORECASE):
                    if Dataset.ASSET.value not in detected:
                        detected.add(Dataset.ASSET.value)

        # Pattern 3: "commonly associated with CWEs" → CAPEC links CWEs to ATT&CK (Q71)
        # "commonly used to exploit" → CAPEC (Q74)
        # "top attack techniques" → CAPEC (Q90: techniques are from ATT&CK, but attack patterns link them)
        # "top [number] CVEs" → CAPEC + CWE + ATT&CK (Q90: comprehensive attack context)
        has_commonly_associate = bool(
            re.search(
                r"\b(commonly\s+associated|commonly\s+used|commonly\s+exploit|top\s+attack\s+techniques)",
                question_lower,
                re.IGNORECASE,
            )
        )
        has_top_cves = bool(
            re.search(r"\btop\s+\d+\s+cves?", question_lower, re.IGNORECASE)
        )
        if has_commonly_associate:
            # If CWE and ATT&CK are both detected, add CAPEC (CAPEC is the bridge)
            if Dataset.CWE.value in detected and Dataset.ATTACK.value in detected:
                if Dataset.CAPEC.value not in detected:
                    detected.add(Dataset.CAPEC.value)
            # "top attack techniques" with workforce → add CAPEC (Q90)
            if re.search(r"top\s+attack\s+techniques", question_lower, re.IGNORECASE):
                if (
                    Dataset.ATTACK.value in detected
                    and Dataset.CAPEC.value not in detected
                ):
                    detected.add(Dataset.CAPEC.value)
                # Also add CWE and CVE for comprehensive attack context
                if Dataset.CWE.value not in detected:
                    detected.add(Dataset.CWE.value)
                if Dataset.CVE.value not in detected:
                    detected.add(Dataset.CVE.value)
        if has_top_cves:
            # "top [number] CVEs" with workforce → comprehensive attack context (Q90)
            # CVE → CWE → CAPEC → ATT&CK
            if Dataset.CVE.value in detected:
                if Dataset.CWE.value not in detected:
                    detected.add(Dataset.CWE.value)
                if Dataset.CAPEC.value not in detected:
                    detected.add(Dataset.CAPEC.value)
                if Dataset.ATTACK.value not in detected:
                    detected.add(Dataset.ATTACK.value)

        # Pattern 4: "[attack type] attacks" without explicit CAPEC mention → CAPEC (Q81, Q86, Q87)
        # "injection attacks", "buffer overflow attacks", "web application attacks"
        # "defending against [attack type] attacks" → CAPEC + ATT&CK (even if no other datasets detected yet)
        has_attack_type_attacks = bool(
            re.search(
                r"\b(injection|buffer overflow|web application|sql injection|xss|cross-site scripting)\s+attacks?",
                question_lower,
                re.IGNORECASE,
            )
        )
        # Also check for "defending against [attack type]" (without "attacks" suffix)
        has_defending_against_attack = bool(
            re.search(
                r"\b(defending|defend|against)\s+(injection|buffer overflow|web application|sql injection|xss|cross-site scripting)",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_attack_type_attacks or has_defending_against_attack:
            # Always add CAPEC for attack type mentions (attack patterns describe these attacks)
            if Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)
            # If defending against attacks, also add ATT&CK (Q87: techniques are used in attacks)
            if has_defending_against_attack or has_attack_type_attacks:
                if Dataset.ATTACK.value not in detected:
                    detected.add(Dataset.ATTACK.value)
            # If CWE is detected (weaknesses) or ATT&CK is detected (techniques), ensure CAPEC is there
            if Dataset.CWE.value in detected or Dataset.ATTACK.value in detected:
                if Dataset.CAPEC.value not in detected:
                    detected.add(Dataset.CAPEC.value)
            # Also ensure CAPEC if NICE is detected (workforce defending against attacks)
            if Dataset.NICE.value in detected:
                if Dataset.CAPEC.value not in detected:
                    detected.add(Dataset.CAPEC.value)

        # Pattern 5: "attack complexity" → CAPEC (CAPEC has complexity metrics) (Q92)
        has_attack_complexity = bool(
            re.search(
                r"\battack\s+complexity|correlation.*attack|complexity.*attack",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_attack_complexity:
            # If ATT&CK is detected, add CAPEC (CAPEC has complexity ratings)
            if Dataset.ATTACK.value in detected and Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)

        # Pattern 6: "defense strategy for [attack]" → CAPEC (Q86, Q94, Q99)
        # "full defense strategy", "defense-in-depth strategy", "mitigations for [attack]"
        # "across all frameworks" or "all available frameworks" → indicates multi-dataset query
        has_defense_strategy = bool(
            re.search(
                r"\b(defense\s+strategy|defense-in-depth|full\s+defense|defense.*strategy|mitigations?\s+for.*attack|across\s+all\s+frameworks|all\s+available\s+frameworks)",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_defense_strategy:
            # If CWE or ATT&CK is detected, add CAPEC (attack patterns are part of defense strategy)
            if (
                Dataset.CWE.value in detected or Dataset.ATTACK.value in detected
            ) and Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)
            # If "across all frameworks" or "all available frameworks" is mentioned, add multiple datasets (Q86, Q99)
            if re.search(
                r"\b(across\s+all|all\s+available)\s+frameworks",
                question_lower,
                re.IGNORECASE,
            ):
                # Q99: "all available frameworks" → CVE + CWE + CAPEC + ATT&CK + NICE + MITIGATION
                if Dataset.CVE.value not in detected:
                    detected.add(Dataset.CVE.value)
                if Dataset.CWE.value not in detected:
                    detected.add(Dataset.CWE.value)
                if Dataset.ATTACK.value not in detected:
                    detected.add(Dataset.ATTACK.value)
                if Dataset.CAPEC.value not in detected:
                    detected.add(Dataset.CAPEC.value)
                if Dataset.MITIGATION.value not in detected:
                    detected.add(Dataset.MITIGATION.value)
                if Dataset.NICE.value not in detected:
                    detected.add(Dataset.NICE.value)
            # Always add CAPEC for defense-in-depth/defense strategy queries
            if Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)

        # ROOT CAUSE 7: Workforce + ATT&CK queries
        # "workforce roles" or "work roles" that "work with" or mention "techniques" → Should detect NICE
        # Pattern: "workforce roles" + "techniques" → NICE + ATT&CK
        has_workforce_mention = bool(
            re.search(
                r"\b(workforce\s+roles?|work\s+roles?|roles?.*workforce)\b",
                question_lower,
                re.IGNORECASE,
            )
        )

        has_technique_mention = bool(
            re.search(
                r"\b(technique|techniques|tactic|tactics|persistence|injection|phishing)\s+(technique|techniques|attack)?",
                question_lower,
                re.IGNORECASE,
            )
        )

        # If question mentions workforce roles AND techniques/attacks, add NICE
        if has_workforce_mention and (
            has_technique_mention or Dataset.ATTACK.value in detected
        ):
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)

        # ROOT CAUSE 8: Knowledge/defense context with attack types
        # "knowledge required for defending against X attacks" → Should detect CWE when X is a weakness type
        # Pattern: "knowledge|defending|defense" + "injection|sql injection|buffer overflow|XSS" → CWE
        weakness_type_keywords = [
            r"\binjection\b",
            r"\bsql\s+injection\b",
            r"\bbuffer\s+overflow\b",
            r"\bxss\b",
            r"\bcross-site\s+scripting\b",
        ]

        has_defense_knowledge_context = bool(
            re.search(
                r"\b(knowledge|required for|defending|defense|mitigation|mitigations)\b",
                question_lower,
                re.IGNORECASE,
            )
        )

        has_weakness_type = any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in weakness_type_keywords
        )

        # If question mentions defense/knowledge AND weakness types, add CWE
        if has_defense_knowledge_context and has_weakness_type:
            if Dataset.CWE.value not in detected:
                detected.add(Dataset.CWE.value)

        # ROOT CAUSE 8.5: Weakness types in vulnerability queries
        # "buffer overflow vulnerabilities", "SQL injection CVEs" → Should detect CWE
        # Pattern: weakness type keywords + "vulnerability|CVE|vulnerabilities" → CWE
        has_vulnerability_context = bool(
            re.search(
                r"\b(vulnerabilit|vulnerabilities|cve|cves)\b",
                question_lower,
                re.IGNORECASE,
            )
        )

        # If question mentions vulnerabilities/CVEs AND weakness types, add CWE
        # (because vulnerabilities are linked to weaknesses via HAS_WEAKNESS)
        if has_vulnerability_context and has_weakness_type:
            if Dataset.CWE.value not in detected:
                detected.add(Dataset.CWE.value)

        # ROOT CAUSE 9: Mitigation queries spanning multiple datasets
        # "mitigation nodes appear in more than one dataset (CWE, CAPEC, ATT&CK)" → Detect all mentioned
        # Pattern: "mitigation" + explicit dataset mentions → Detect those datasets
        if re.search(r"\bmitigation", question_lower, re.IGNORECASE):
            # Check for explicit dataset mentions in parentheses or list format
            if re.search(r"\bcwe\b", question_lower, re.IGNORECASE):
                detected.add(Dataset.CWE.value)
            if re.search(r"\bcapec\b", question_lower, re.IGNORECASE):
                detected.add(Dataset.CAPEC.value)
            # ATT&CK can be mentioned as "ATT&CK", "ATTACK", or implicit via "tactics", "techniques", "Mitre"
            # Q54: "ATT&CK" in parentheses - check for explicit ATT&CK/ATTACK mention
            # Note: "ATT&CK" may be encoded as "att&ck" or "att&amp;ck" or just "attack"
            # BUT: "attack patterns" refers to CAPEC, not ATT&CK - must check for explicit ATT&CK mention
            has_explicit_attack = bool(
                re.search(
                    r"\batt\s*&\s*ck\b|\batt\s*&amp\s*ck\b|\bmitre\b|\bt\d{4}\b|\btactic|\btechnique",
                    question_lower,
                    re.IGNORECASE,
                )
            )
            # Only add ATT&CK if explicitly mentioned (not just "attack" in "attack patterns")
            if has_explicit_attack:
                detected.add(Dataset.ATTACK.value)
            if re.search(r"\bcve\b|\bvulnerabilit", question_lower, re.IGNORECASE):
                detected.add(Dataset.CVE.value)

        # ROOT CAUSE 10: Roles/workforce queries without explicit framework mention
        # "Which roles involve threat hunting?" → Should detect NICE (roles = work roles)
        # "Which roles are involved in mitigating privilege escalation?" → NICE + ATT&CK
        has_roles_mention = bool(
            re.search(
                r"\broles?\s+(involve|involved|work|required|activities)",
                question_lower,
                re.IGNORECASE,
            )
        )
        has_threat_hunting = bool(
            re.search(r"\bthreat\s+hunting", question_lower, re.IGNORECASE)
        )
        has_privilege_escalation = bool(
            re.search(r"\bprivilege\s+escalation", question_lower, re.IGNORECASE)
        )
        has_mitigating_context = bool(
            re.search(r"\bmitigat", question_lower, re.IGNORECASE)
        )

        # If question mentions roles + threat hunting, add NICE
        if has_roles_mention and has_threat_hunting:
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)

        # If question mentions roles + privilege escalation + mitigating, add NICE + ATT&CK
        # Q69: "roles are involved in mitigating privilege escalation"
        # Q80: "tasks required to mitigate privilege escalation attacks" → NICE + ATT&CK
        if has_roles_mention and has_privilege_escalation and has_mitigating_context:
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)
            if Dataset.ATTACK.value not in detected:
                detected.add(Dataset.ATTACK.value)

        # ROOT CAUSE 15: Tactic name detection (privilege escalation, persistence, lateral movement, etc.)
        # Tactic names are ATT&CK-specific indicators
        # Q80: "privilege escalation attacks" → ATT&CK (privilege escalation is a tactic)
        # Also: "tasks required to mitigate [tactic] attacks" → NICE + ATT&CK
        tactic_names = [
            r"\bprivilege\s+escalation\b",
            r"\bpersistence\b",
            r"\blateral\s+movement\b",
            r"\binitial\s+access\b",
            r"\bexecution\b",
            r"\bdefense\s+evasion\b",
            r"\bcredential\s+access\b",
            r"\bdiscovery\b",
            r"\bcollection\b",
            r"\bcommand\s+and\s+control\b",
            r"\bexfiltration\b",
            r"\bimpact\b",
        ]
        has_tactic_name = any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in tactic_names
        )
        if has_tactic_name:
            # Tactic names always indicate ATT&CK
            if Dataset.ATTACK.value not in detected:
                detected.add(Dataset.ATTACK.value)
            # If workforce/tasks are mentioned with tactics, also add NICE
            if (
                has_roles_mention
                or re.search(
                    r"\btasks?|workforce|skills?", question_lower, re.IGNORECASE
                )
            ) and Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)

        # ROOT CAUSE 16: "attack techniques" explicit mention → ATT&CK (Q97)
        # "current attack techniques", "top attack techniques" → ATT&CK
        has_attack_techniques = bool(
            re.search(
                r"\b(attack\s+techniques?|current\s+attack|top\s+attack|techniques?\s+attack)",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_attack_techniques:
            if Dataset.ATTACK.value not in detected:
                detected.add(Dataset.ATTACK.value)
            # If workforce is mentioned with attack techniques, also add NICE
            if (
                re.search(r"\bworkforce|skills?", question_lower, re.IGNORECASE)
                and Dataset.NICE.value not in detected
            ):
                detected.add(Dataset.NICE.value)
            # Also add CVE for comprehensive attack context
            if Dataset.CVE.value not in detected:
                detected.add(Dataset.CVE.value)

        # ROOT CAUSE 17: Workforce expertise + vulnerabilities → ATT&CK (Q100)
        # "workforce expertise", "required workforce" with vulnerabilities → CVE + ATT&CK + NICE
        has_workforce_expertise = bool(
            re.search(
                r"\b(workforce\s+expertise|required\s+workforce|workforce\s+competenc|correlation.*workforce)",
                question_lower,
                re.IGNORECASE,
            )
        )
        has_vulnerability_severity = bool(
            re.search(
                r"\bvulnerabilit.*severity|severity.*vulnerabilit|correlation.*vulnerabilit",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_workforce_expertise and (
            Dataset.CVE.value in detected or has_vulnerability_severity
        ):
            if Dataset.CVE.value not in detected:
                detected.add(Dataset.CVE.value)
            if Dataset.ATTACK.value not in detected:
                detected.add(Dataset.ATTACK.value)
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)

        # ROOT CAUSE 23: Workforce training/requirements → NICE (Q89, Q92)
        # "workforce training requirements", "required workforce expertise" → NICE
        # "correlation between [datasets] and required workforce [X]" → NICE
        has_workforce_training = bool(
            re.search(
                r"\b(workforce\s+training|training\s+requirements?|workforce\s+requirements?)",
                question_lower,
                re.IGNORECASE,
            )
        )
        # "correlation between [X] and required workforce [Y]" → NICE (Q92)
        has_correlation_workforce = bool(
            re.search(
                r"\bcorrelation\s+between.*required\s+workforce|correlation.*workforce\s+expertise",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_workforce_training or has_correlation_workforce:
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)
            # If ATT&CK or CAPEC is mentioned, ensure they're detected too
            if re.search(
                r"\b(techniques?|attack|complexity)", question_lower, re.IGNORECASE
            ):
                if Dataset.ATTACK.value not in detected:
                    detected.add(Dataset.ATTACK.value)
            if re.search(
                r"\b(attack\s+complexity|capec)", question_lower, re.IGNORECASE
            ):
                if Dataset.CAPEC.value not in detected:
                    detected.add(Dataset.CAPEC.value)
            if re.search(
                r"\b(weaknesses?|social\s+engineering)", question_lower, re.IGNORECASE
            ):
                if Dataset.CWE.value not in detected:
                    detected.add(Dataset.CWE.value)

        # Also catch "roles are involved in [action]" pattern
        if re.search(r"\broles?\s+are\s+involved\s+in", question_lower, re.IGNORECASE):
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)
            # If it mentions privilege escalation (ATT&CK tactic) or other ATT&CK concepts, add ATT&CK
            if has_privilege_escalation or re.search(
                r"\btactic|\btechnique", question_lower, re.IGNORECASE
            ):
                if Dataset.ATTACK.value not in detected:
                    detected.add(Dataset.ATTACK.value)

        # ROOT CAUSE 21: "threat landscape" → ATT&CK (Q97)
        # "current threat landscape" with workforce/skills gaps → ATT&CK + NICE + CVE
        has_threat_landscape = bool(
            re.search(
                r"\b(threat\s+landscape|current\s+threat)",
                question_lower,
                re.IGNORECASE,
            )
        )
        has_skills_gaps = bool(
            re.search(r"\bskills?\s+gaps", question_lower, re.IGNORECASE)
        )
        if has_threat_landscape and (
            has_skills_gaps or re.search(r"\bworkforce", question_lower, re.IGNORECASE)
        ):
            # Threat landscape queries should include ATT&CK (techniques are part of threat landscape)
            if Dataset.ATTACK.value not in detected:
                detected.add(Dataset.ATTACK.value)
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)
            if Dataset.CVE.value not in detected:
                detected.add(Dataset.CVE.value)

        # ROOT CAUSE 22: "all framework relationships" → All datasets including ATT&CK (Q99)
        # "all framework relationships" or "all framework" → comprehensive multi-dataset query
        if re.search(
            r"\ball\s+framework\s+relationships?", question_lower, re.IGNORECASE
        ):
            # Ensure all major datasets are detected
            if Dataset.CVE.value not in detected:
                detected.add(Dataset.CVE.value)
            if Dataset.CWE.value not in detected:
                detected.add(Dataset.CWE.value)
            if Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)
            if Dataset.ATTACK.value not in detected:
                detected.add(Dataset.ATTACK.value)
            if Dataset.MITIGATION.value not in detected:
                detected.add(Dataset.MITIGATION.value)
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)

        # ROOT CAUSE 24: "ALL DATASETS" detection (Q88, Q93, Q96, Q101)
        # "complete attack surface", "complete threat landscape", "all relevant frameworks",
        # "all available attack frameworks", "all cybersecurity frameworks"
        # "including vulnerabilities, weaknesses, attack patterns, techniques, and defenses" → all datasets
        has_all_datasets_indicators = bool(
            re.search(
                r"\b(complete\s+attack\s+surface|complete\s+threat\s+landscape|"
                r"all\s+relevant\s+frameworks|all\s+available\s+attack\s+frameworks|"
                r"all\s+cybersecurity\s+frameworks|across\s+all.*framework|"
                r"vulnerabilities?.*weaknesses?.*attack\s+patterns?.*techniques?.*defenses?)",
                question_lower,
                re.IGNORECASE,
            )
        )
        # "attack chain" with "all available" or "using all" → all datasets (Q96)
        has_attack_chain_all = bool(
            re.search(
                r"\battack\s+chain.*(all\s+available|using\s+all|all\s+attack\s+frameworks)",
                question_lower,
                re.IGNORECASE,
            )
        )
        # "crosswalk relationships" with "all" or "effective threat hunting" → all datasets (Q101)
        has_crosswalk_all = bool(
            re.search(
                r"\bcrosswalk\s+relationships?.*(all|critical|effective)",
                question_lower,
                re.IGNORECASE,
            )
        )
        # "threat hunting" across frameworks → all datasets (Q101)
        has_threat_hunting_all = bool(
            re.search(
                r"\bthreat\s+hunting.*(across|all|framework)",
                question_lower,
                re.IGNORECASE,
            )
        )
        # "across all cybersecurity frameworks" → all datasets (Q101)
        has_across_all_cybersecurity = bool(
            re.search(
                r"\bacross\s+all\s+cybersecurity\s+frameworks",
                question_lower,
                re.IGNORECASE,
            )
        )
        if (
            has_all_datasets_indicators
            or has_attack_chain_all
            or has_crosswalk_all
            or has_threat_hunting_all
            or has_across_all_cybersecurity
        ):
            # Add ALL major datasets
            all_datasets = [
                Dataset.CVE.value,
                Dataset.CWE.value,
                Dataset.CAPEC.value,
                Dataset.ATTACK.value,
                Dataset.MITIGATION.value,
                Dataset.NICE.value,
                Dataset.DCWF.value,
            ]
            for ds in all_datasets:
                if ds not in detected:
                    detected.add(ds)

        # Also catch standalone "roles" queries (Q63: "Which roles involve...")
        # Generic "roles" without explicit "workforce" or "work role" mention
        if re.search(r"\bwhich\s+roles?|roles?\s+which", question_lower, re.IGNORECASE):
            if (
                Dataset.NICE.value not in detected
                and Dataset.DCWF.value not in detected
            ):
                # Generic role query should detect NICE (most common)
                detected.add(Dataset.NICE.value)

        # ROOT CAUSE 11: DCWF tasks + CAPEC queries
        # "What DCWF tasks connect to CAPEC patterns?" → Should detect both DCWF and NICE
        # (DCWF tasks are accessed via NICE WorkRole nodes)
        has_dcwf_tasks = bool(
            re.search(r"\bdcwf\s+tasks?", question_lower, re.IGNORECASE)
        )
        if has_dcwf_tasks and Dataset.CAPEC.value in detected:
            if Dataset.DCWF.value not in detected:
                detected.add(Dataset.DCWF.value)
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)

        # ROOT CAUSE 12: WorkRoles defending against vulnerabilities
        # "WorkRoles that defend against injection vulnerabilities" → CWE + NICE + ATT&CK
        # "WorkRoles that work with vulnerabilities" → CVE + NICE + ATT&CK (Q65)
        has_workrole_defending = bool(
            re.search(
                r"\b(workroles?|work\s+roles?)\s+that\s+(defend|defending|required|work\s+with)",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_workrole_defending and has_weakness_type:
            if Dataset.CWE.value not in detected:
                detected.add(Dataset.CWE.value)
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)
            # If defending against vulnerabilities that map to techniques, add ATT&CK
            if re.search(r"\bvulnerabilit", question_lower, re.IGNORECASE):
                detected.add(Dataset.ATTACK.value)
        elif has_workrole_defending and re.search(
            r"\bvulnerabilit", question_lower, re.IGNORECASE
        ):
            # Q65: "WorkRoles that work with vulnerabilities" → should detect ATT&CK (vulnerabilities link to techniques)
            if Dataset.NICE.value not in detected:
                detected.add(Dataset.NICE.value)
            if Dataset.CVE.value not in detected:
                detected.add(Dataset.CVE.value)
            if Dataset.ATTACK.value not in detected:
                detected.add(Dataset.ATTACK.value)

        # ROOT CAUSE 13: "vulnerabilities and attack patterns" → CVE + CAPEC + CWE
        # "Which mitigations address both vulnerabilities and attack patterns?" → CVE + CAPEC + CWE
        # Must be careful: "address both" pattern should NOT add ATT&CK (only for explicit ATT&CK mentions)
        has_both_vuln_and_attack = bool(
            re.search(
                r"\b(both|and)\s+(vulnerabilities?|vuln).*attack\s+patterns?|attack\s+patterns?.*(and|both).*vulnerabilities?",
                question_lower,
                re.IGNORECASE,
            )
        )
        if has_both_vuln_and_attack and re.search(
            r"\bmitigation", question_lower, re.IGNORECASE
        ):
            if Dataset.CVE.value not in detected:
                detected.add(Dataset.CVE.value)
            if Dataset.CAPEC.value not in detected:
                detected.add(Dataset.CAPEC.value)
            if Dataset.CWE.value not in detected:
                detected.add(Dataset.CWE.value)
            # Remove ATT&CK if it was incorrectly added (Q57: should not have ATT&CK)
            if Dataset.ATTACK.value in detected and not re.search(
                r"\battack\b|\bmitre\b|\btactic|\btechnique",
                question_lower,
                re.IGNORECASE,
            ):
                detected.discard(Dataset.ATTACK.value)

        return sorted(list(detected)) if detected else []

    def _detect_crosswalks(self, question_lower: str, datasets: List[str]) -> List[str]:
        """Detect crosswalk relationships."""
        crosswalks = []

        # Direct pattern matching
        for crosswalk, patterns in self.CROSSWALK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    crosswalks.append(crosswalk)
                    break

        # Infer from datasets (if multiple datasets mentioned, likely crosswalk)
        if len(datasets) >= 2:
            dataset_set = set(datasets)
            if Dataset.CVE.value in dataset_set and Dataset.CWE.value in dataset_set:
                if "CVE<->CWE" not in crosswalks:
                    crosswalks.append("CVE<->CWE")
            if Dataset.CVE.value in dataset_set and Dataset.ASSET.value in dataset_set:
                if "CVE<->Asset" not in crosswalks:
                    crosswalks.append("CVE<->Asset")
            if Dataset.CVE.value in dataset_set and Dataset.ATTACK.value in dataset_set:
                if "CVE<->ATT&CK" not in crosswalks:
                    crosswalks.append("CVE<->ATT&CK")
            if (
                Dataset.CAPEC.value in dataset_set
                and Dataset.ATTACK.value in dataset_set
            ):
                if "CAPEC<->ATT&CK" not in crosswalks:
                    crosswalks.append("CAPEC<->ATT&CK")
            if (
                Dataset.CWE.value in dataset_set
                and Dataset.MITIGATION.value in dataset_set
            ):
                if "CWE<->Mitigation" not in crosswalks:
                    crosswalks.append("CWE<->Mitigation")
            if (
                Dataset.CAPEC.value in dataset_set
                and Dataset.CATEGORY.value in dataset_set
            ):
                if "CAPEC<->Category" not in crosswalks:
                    crosswalks.append("CAPEC<->Category")
            if Dataset.NICE.value in dataset_set and Dataset.DCWF.value in dataset_set:
                if "NICE<->DCWF" not in crosswalks:
                    crosswalks.append("NICE<->DCWF")

        return sorted(list(set(crosswalks)))

    def _detect_complexity(
        self, question_lower: str, datasets: List[str], crosswalks: List[str]
    ) -> str:
        """Detect complexity level.

        Complexity heuristics:
        - Easy: Single dataset, OR simple direct traversal with specific entity ID
        - Medium: Crosswalks requiring 2-hop traversal, semantic search, or filtering
        - Hard: 4+ datasets, OR multiple crosswalks + 3+ datasets, OR explicit hard patterns
        """
        # Check explicit complexity patterns first
        for complexity, patterns in self.COMPLEXITY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return complexity

        # Refined heuristics based on datasets and crosswalks
        num_datasets = len(datasets)
        num_crosswalks = len(crosswalks)

        # ROOT CAUSE 20: Check hard patterns FIRST (before dataset-based logic)
        # Complete defense strategy, attack chain, statistical queries
        # These patterns should be checked early, regardless of dataset count
        hard_indicators = [
            r"complete\s+defense-in-depth",  # Q99
            r"defense-in-depth\s+strategy",  # Q99: "defense-in-depth strategy"
            r"all\s+framework\s+relationships",  # Q99: "all framework relationships" (variant)
            r"attack\s+chain\s+from.*to",  # Q96: "attack chain from social engineering to data exfiltration"
            r"attack\s+path\s+from.*to",  # Q73: "attack path from CVE to ATT&CK technique"
            r"highest\s+overlap\s+between",  # Q95: "highest overlap between NICE and DCWF"
            r"skills?\s+gaps",  # Q97: "skills gaps"
            r"all\s+available\s+frameworks",  # Q99: comprehensive query
            r"related.*and.*and",  # Q82: "related weaknesses, attack patterns, techniques, and roles"
            # ROOT CAUSE 24: "ALL DATASETS" patterns → hard
            r"complete\s+attack\s+surface",  # Q88
            r"complete\s+threat\s+landscape",  # Q93
            r"all\s+relevant\s+frameworks",  # Q93
            r"all\s+available\s+attack\s+frameworks",  # Q96
            r"attack\s+chain.*all\s+available",  # Q96: "attack chain...using all available"
            r"all\s+cybersecurity\s+frameworks",  # Q101
            r"across\s+all\s+cybersecurity\s+frameworks",  # Q101
            r"crosswalk\s+relationships?.*(critical|effective|all)",  # Q101
            r"threat\s+hunting.*across\s+all",  # Q101
        ]
        if any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in hard_indicators
        ):
            return "hard"

        # 5+ datasets → hard (check this BEFORE crosswalk logic to avoid early returns)
        if num_datasets >= 5:
            return "hard"

        # Hard: Multiple crosswalks with 3+ datasets OR explicit multi-hop indicators
        # BUT: 4 datasets without explicit hard indicators → medium (Q54, Q65)
        if num_crosswalks >= 2 and num_datasets >= 3:
            # Check for explicit multi-hop indicators
            multi_hop_patterns = [
                r"via.*and.*through",
                r"inferred.*through",
                r"complete.*chain",
                r"path from.*to.*to",
                r"multi.*hop",
                r"complete.*attack",
            ]
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in multi_hop_patterns
            ):
                return "hard"

            # ROOT CAUSE 18: Aggregation + multi-hop → hard (Q71, Q74, Q75)
            # "commonly associated", "commonly used", "most prevalent" require aggregation across datasets
            aggregation_patterns = [
                r"commonly\s+(associated|used)",
                r"most\s+prevalent",
                r"most\s+common",
                r"highest\s+(overlap|count|number)",
            ]
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in aggregation_patterns
            ):
                return "hard"

            # ROOT CAUSE 19: Multi-hop workforce queries → hard (Q76, Q80)
            # "responsible for mitigating", "tasks required to mitigate", "skills gaps"
            workforce_multi_hop = [
                r"responsible\s+for\s+mitigating",
                r"tasks?\s+required\s+to\s+mitigate",
                r"skills?\s+gaps",
            ]
            # Statistical/mapping patterns (Q84)
            statistical_patterns = [
                r"mapped\s+to",
                r"align\s+with",
                r"relate\s+to",
                r"relationship\s+between",
            ]
            # Comprehensive coverage (Q98)
            comprehensive_patterns = [
                r"most\s+comprehensive\s+mitigation",
                r"comprehensive\s+(mitigation|coverage)",
            ]
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in workforce_multi_hop
                + statistical_patterns
                + comprehensive_patterns
            ):
                return "hard"

            return "medium"  # 2+ crosswalks with 3 datasets → medium

        # Easy: Single dataset (no crosswalks)
        # BUT: Single dataset with semantic search + traversal might be medium (Q62, Q63)
        # BUT: Single dataset queries should never be hard (hard requires multi-dataset complexity)
        if num_datasets == 1:
            # Check for medium indicators that suggest semantic search complexity
            medium_semantic_indicators = [
                r"\broles?\s+involve.*activities?",  # Q63: semantic search for threat hunting
                r"\btasks?\s+(are\s+)?associated\s+with.*assessment",  # Q62: "tasks are associated with vulnerability assessment"
                r"\btasks?\s+(are\s+)?associated\s+with.*\w+",  # Q62: tasks [are] associated with [semantic term] → medium (requires semantic matching)
            ]
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in medium_semantic_indicators
            ):
                return "medium"
            return "easy"

        # ROOT CAUSE 25: Hard pattern check BEFORE dataset-based logic
        # Check for hard patterns that apply regardless of crosswalk count
        # This must happen before the crosswalk/dataset checks to catch all hard queries
        hard_patterns_anywhere = [
            # Aggregation patterns (Q75, Q87, Q90, Q98)
            r"most\s+prevalent",
            r"most\s+critical",
            r"most\s+comprehensive",
            r"commonly\s+(associated|used)",
            r"most\s+common",
            r"highest\s+(overlap|count|number)",
            # Workforce multi-hop (Q80, Q87, Q90)
            r"tasks?\s+required\s+to\s+mitigate",
            r"tasks?\s+are\s+required\s+to\s+mitigate",  # Q80: more specific
            r"responsible\s+for\s+mitigating",
            r"skills?\s+gaps",
            r"workforce\s+skills?\s+(needed|required)",
            r"most\s+critical\s+workforce\s+skills",  # Q90: more specific
            # Defense multi-hop (Q78, Q81, Q87, Q90)
            r"roles?\s+involved\s+in\s+defending",
            r"roles?\s+would\s+be\s+involved\s+in\s+defending",  # Q78: "work roles would be involved in defending"
            r"work\s+roles?\s+would\s+be\s+involved\s+in\s+defending",  # Q78: more specific
            r"would\s+be\s+involved\s+in\s+defending",  # Q78: even more flexible
            r"defending\s+against.*(roles?|workforce|tasks?)",
            r"defend\s+against.*(roles?|workforce|tasks?)",
            r"involved\s+in\s+defending",  # Q78: core pattern
            # Statistical/mapping patterns (Q84, Q89, Q91, Q83)
            r"relate\s+to",
            r"relationship\s+between",
            r"mapped\s+to",
            r"align\s+with",
            r"alignment\s+with",
            # Comprehensive coverage (Q98)
            r"comprehensive\s+(mitigation|coverage)",
            r"most\s+comprehensive\s+mitigation",  # Q98: more specific
        ]
        if any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in hard_patterns_anywhere
        ):
            # If we have 2+ datasets with hard patterns, it's hard
            if num_datasets >= 2:
                return "hard"

        # Medium/Hard check: Multiple crosswalks OR 3+ datasets
        # BUT: 3-4 datasets with simple relationships might still be medium (not hard)
        if num_crosswalks >= 2:
            # Check for hard patterns even with 2 crosswalks
            # ROOT CAUSE 18 & 19: Aggregation and workforce multi-hop patterns
            aggregation_patterns = [
                r"commonly\s+(associated|used)",
                r"most\s+prevalent",
                r"most\s+common",
                r"highest\s+(overlap|count|number)",
            ]
            workforce_multi_hop = [
                r"responsible\s+for\s+mitigating",
                r"tasks?\s+required\s+to\s+mitigate",
                r"skills?\s+gaps",
            ]
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in aggregation_patterns + workforce_multi_hop
            ):
                return "hard"
            return "medium"
        if num_datasets >= 3:
            # Check if it's a complex multi-hop or just simple multi-dataset query
            # Hard: explicit multi-hop indicators OR 4+ datasets with hard patterns
            if num_datasets >= 4:
                # Check for explicit hard patterns before defaulting to medium
                multi_hop_patterns = [
                    r"inferred\s+through",
                    r"complete.*chain",
                    r"across.*all.*framework",
                    r"all\s+available\s+frameworks",
                    r"correlation",
                    r"attack\s+surface",
                    r"related.*and.*and",  # "related weaknesses, attack patterns, techniques, and roles"
                    r"what\s+are\s+the\s+related",  # Q82: "what are the related weaknesses"
                ]
                aggregation_patterns = [
                    r"commonly\s+(associated|used)",
                    r"most\s+prevalent",
                    r"most\s+common",
                    r"highest\s+(overlap|count|number)",
                    r"most\s+critical",
                    r"most\s+comprehensive",
                ]
                workforce_multi_hop = [
                    r"responsible\s+for\s+mitigating",
                    r"tasks?\s+required\s+to\s+mitigate",
                    r"skills?\s+gaps",
                ]
                # ROOT CAUSE 20: Statistical/correlation queries → hard (Q95, Q100)
                statistical_patterns = [
                    r"overlap\s+between",
                    r"correlation\s+between",
                    r"highest\s+overlap",
                    r"relate\s+to",
                    r"relationship\s+between",
                    r"mapped\s+to",
                    r"align\s+with",
                ]
                defense_patterns = [
                    r"roles?\s+involved\s+in\s+defending",
                    r"defending\s+against",
                ]
                # Comprehensive mitigation coverage (Q98)
                comprehensive_patterns = [
                    r"most\s+comprehensive\s+mitigation",
                    r"comprehensive\s+(mitigation|coverage)",
                ]
                if any(
                    re.search(pattern, question_lower, re.IGNORECASE)
                    for pattern in multi_hop_patterns
                    + aggregation_patterns
                    + workforce_multi_hop
                    + statistical_patterns
                    + defense_patterns
                    + comprehensive_patterns
                ):
                    return "hard"
                # 4+ datasets without explicit hard indicators → still medium (Q54, Q65)
                return "medium"
            # 3 datasets with hard patterns → hard
            aggregation_patterns = [
                r"commonly\s+(associated|used)",
                r"most\s+prevalent",
                r"most\s+critical",
                r"attack\s+path\s+from",  # Q73: "attack path from CVE to ATT&CK"
            ]
            workforce_multi_hop = [
                r"responsible\s+for\s+mitigating",
                r"tasks?\s+required\s+to\s+mitigate",
            ]
            statistical_patterns = [
                r"relationship\s+between",
                r"relate\s+to",
            ]
            defense_patterns = [
                r"defending\s+against",
                r"roles?\s+involved\s+in\s+defending",
            ]
            if any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in aggregation_patterns
                + workforce_multi_hop
                + statistical_patterns
                + defense_patterns
            ):
                return "hard"
            # 3 datasets → medium
            return "medium"

        # 2 datasets with 1 crosswalk: Check for hard patterns (Q75, Q91)
        if num_datasets == 2:
            # Aggregation with 2 datasets → hard (Q75)
            if re.search(
                r"most\s+prevalent|most\s+critical", question_lower, re.IGNORECASE
            ):
                return "hard"
            # Statistical relationships with 2 datasets → hard (Q91)
            if re.search(
                r"relate\s+to|relationship\s+between", question_lower, re.IGNORECASE
            ):
                return "hard"
            # Otherwise, 2 datasets with crosswalk → medium
            return "medium"

        # For 2 datasets with 1 crosswalk: Check if it's a simple direct traversal
        if num_datasets == 2 and num_crosswalks == 1:
            # Check for specific entity ID (CVE-XXXX, CWE-XX, CAPEC-XXX, T####)
            has_specific_id = bool(
                re.search(
                    r"\b(cve-\d{4}-\d+|cwe-\d+|capec-\d+|t\d{4}|attack pattern (capec|cwe)-\d+)\b",
                    question_lower,
                    re.IGNORECASE,
                )
            )

            # Simple traversal/filtering patterns that should be EASY
            simple_patterns = [
                r"does (cve|cwe|capec)-\w+ (affect|link|connect)",
                r"(vendor|product).*does.*affect",
                r"mitigations?.*listed for",
                r"labeled.*category",  # Category filter is simple (Q17)
                r"which.*patterns?.*labeled.*category",  # Q17 pattern
                r"fall under.*(tactic|technique)",  # Direct relationship within ATT&CK
                r"associated with.*work role",  # Workforce queries are simple (Q27: "tasks belong to work role")
                r"specialty areas?.*in (dcwf|nice)",  # Simple workforce queries
                r"work roles?.*exist",  # Count queries
                r"tasks?.*fall under",  # Workforce queries
                r"show me.*(tasks?|roles?)",  # Generic workforce list
                r"show me.*vulnerabilities?.*for",  # Simple vendor filter (Q5)
                r"related to.*technique\s+t\d{4}",  # Related to specific technique (Q23)
                r"used by.*attack patterns?",  # Simple reverse traversal (Q26)
            ]

            is_simple_pattern = any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in simple_patterns
            )

            # Workforce crosswalks (NICE↔DCWF) are typically easy
            is_workforce_crosswalk = "NICE<->DCWF" in crosswalks or (
                Dataset.NICE.value in datasets and Dataset.DCWF.value in datasets
            )

            # Check for explicit "both frameworks" or "via" which indicates set intersection (medium)
            has_both_frameworks_mention = bool(
                re.search(
                    r"\bappear in both|both.*framework|via.*crosswalk|via.*dcwf.*nice",
                    question_lower,
                    re.IGNORECASE,
                )
            )

            # Medium indicators: complex semantic search with cross-dataset filtering
            # These require traversing the crosswalk AND applying semantic filters
            medium_indicators = [
                r"\b(affect|affects|affecting)\s+(linux|windows|microsoft|apple|product|systems?).*\s+through",  # Q42 pattern: "affect X through Y"
                r"\blinked to\s+.*(weakness|vulnerabilit).*\s+.*(buffer|overflow|sql|xss)",  # Q43: Semantic weakness matching with descriptive terms
                r"\bmap to\s+.*technique\s+t\d{4}",  # Q45: Mapping to specific technique
                r"\b(which|what)\s+.*(tactics?|techniques?).*\s+used by.*(phishing).*\s+patterns?",  # Q47: Reverse traversal with semantic filtering (phishing-specific)
                r"\b.*tactics?.*used by.*phishing.*attack patterns?",  # Q47: "tactics used by phishing-related CAPEC attacks"
                r"\btactics?.*used by.*phishing",  # Q47: "tactics are used by phishing-related" (more flexible)
                r"\brelated to\s+.*(phishing|web application|persistence).*\s+attacks?",  # Q48: Semantic technique matching
                r"\broles?\s+involve.*activities?",  # Q63: "roles involve [activity]" → medium (semantic search)
                r"\btasks?\s+associated with.*assessment",  # Q62: "tasks associated with vulnerability assessment" → medium
                r"\bappear in both.*framework.*via",  # Q60: "appear in both frameworks via" → medium (set intersection)
            ]

            has_medium_indicators = any(
                re.search(pattern, question_lower, re.IGNORECASE)
                for pattern in medium_indicators
            )

            # Decision logic:
            # 1. Medium indicators → Medium (prioritize over simple patterns when both match)
            # 2. Workforce crosswalks with "both frameworks" or "via" → Medium (set intersection)
            # 3. Workforce crosswalks -> Easy (default for NICE<->DCWF)
            # 4. Simple patterns → Easy (only if no medium indicators)
            # 5. No simple pattern + no medium indicators → Medium (default)
            if has_medium_indicators:
                # Medium indicators take priority (semantic search + crosswalk = medium)
                return "medium"
            elif is_workforce_crosswalk and has_both_frameworks_mention:
                # "appear in both frameworks via" indicates set intersection (medium)
                return "medium"
            elif is_workforce_crosswalk:
                # Default: workforce crosswalks are easy (simple shared WorkRole queries)
                return "easy"
            elif is_simple_pattern:
                # Simple patterns indicate easy traversal, even without specific ID
                return "easy"
            else:
                # Default for other 2 datasets + 1 crosswalk is medium
                return "medium"

        # Default to medium if unclear
        return "medium"

    def _detect_intents(self, question_lower: str) -> List[str]:
        """Detect intent types (can be multiple)."""
        intents = []

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    intents.append(intent.value)
                    break

        # If no intents detected, default to list
        if not intents:
            intents.append(IntentType.LIST.value)

        return sorted(list(set(intents)))

    def _determine_schema_packs(
        self, datasets: List[str], crosswalks: List[str]
    ) -> List[str]:
        """Determine expected schema packs."""
        packs = []

        # Add dataset packs
        for dataset in datasets:
            packs.append(dataset)

        # Add crosswalk packs
        for crosswalk in crosswalks:
            # Convert crosswalk notation to pack name
            pack_name = crosswalk.replace("<->", "_").replace("&", "").upper()
            packs.append(pack_name)

        return sorted(list(set(packs)))

    def _extract_key_properties(
        self, question_lower: str, datasets: List[str], intents: List[str]
    ) -> List[str]:
        """Extract key properties needed for query."""
        properties = []

        # Dataset-specific properties
        if Dataset.CVE.value in datasets:
            if "cvss" in question_lower or IntentType.FILTER.value in intents:
                properties.append("cvss_v31")
            if (
                "severity" in question_lower
                or "critical" in question_lower
                or "high" in question_lower
                or "medium" in question_lower
                or "low" in question_lower
            ):
                properties.append("severity")
            if "2024" in question_lower or "year" in question_lower:
                properties.append("year")

        if Dataset.CWE.value in datasets:
            properties.append("description")

        if (
            Dataset.ASSET.value in datasets
            or "vendor" in question_lower
            or "product" in question_lower
        ):
            properties.append("vendor")
            properties.append("product")

        if Dataset.ATTACK.value in datasets:
            if "tactic" in question_lower:
                properties.append("x_mitre_domains")
            if "platform" in question_lower:
                properties.append("x_mitre_platforms")

        if Dataset.NICE.value in datasets or Dataset.DCWF.value in datasets:
            # Following schema packs: Use COALESCE(wr.work_role, wr.title) for role queries
            # work_role exists on 74/115 nodes (DCWF), title exists on 41/115 nodes (NICE)
            if (
                "work role" in question_lower
                or "administrator" in question_lower
                or "analyst" in question_lower
            ):
                properties.append("work_role")  # For DCWF roles (74 nodes)
                properties.append("title")  # For NICE roles (41 nodes)
                properties.append("text")  # Alternative property
            if "task" in question_lower:
                properties.append("title")  # Task.title or Task.text per schema packs
                properties.append("text")
            if "skill" in question_lower:
                properties.append("title")  # Skill.title or Skill.text per schema packs
                properties.append("text")
            if "knowledge" in question_lower:
                properties.append(
                    "title"
                )  # Knowledge.title or Knowledge.text per schema packs
                properties.append("text")
            if "ability" in question_lower:
                properties.append(
                    "description"
                )  # Ability.description or Ability.text per schema packs
                properties.append("text")
            if "dcwf" in question_lower or re.search(r"work role \d+", question_lower):
                properties.append("dcwf_code")  # For DCWF-aligned roles (74 nodes)

        return sorted(list(set(properties)))

    def should_use_rag(
        self,
        question: str,
        classification_result: Optional[ClassificationResult] = None,
    ) -> bool:
        """Determine if question should use RAG (vector similarity search) instead of Cypher.

        RAG is used for:
        - Similarity queries ("similar to", "like", "most similar")
        - Conceptual queries without exact keywords
        - Natural language descriptions that need semantic understanding

        Args:
            question: Natural language question
            classification_result: Optional ClassificationResult (if already classified)

        Returns:
            True if RAG should be used, False otherwise
        """
        if classification_result is None:
            classification_result = self.classify(question)

        question_lower = question.lower()

        # Check for explicit similarity search intent
        if IntentType.SIMILARITY_SEARCH.value in classification_result.intent_types:
            return True

        # Check for conceptual queries (semantic search without exact keywords)
        # These are queries that would benefit from semantic similarity
        conceptual_patterns = [
            r"mechanisms being bypassed",
            r"where attackers can",
            r"that involve",
            r"related to.*conceptually",
            r"conceptually.*related",
        ]

        has_conceptual_indicator = any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in conceptual_patterns
        )

        # If it's a conceptual query and NOT an exact ID match, use RAG
        if has_conceptual_indicator:
            # Check if it's NOT an exact ID match (exact IDs should use Cypher)
            has_exact_id = bool(
                re.search(
                    r"\b(CVE|CWE|CAPEC|T\d{4})-\d+", question_lower, re.IGNORECASE
                )
            )
            if not has_exact_id:
                return True

        # Check for natural language descriptions that need semantic understanding
        # e.g., "vulnerabilities where attackers can execute code remotely"
        natural_language_patterns = [
            r"where.*can.*execute",
            r"that allow.*to",
            r"enabling.*to",
            r"permitting.*to",
        ]

        has_natural_language = any(
            re.search(pattern, question_lower, re.IGNORECASE)
            for pattern in natural_language_patterns
        )

        # If natural language query and semantic_search intent, use RAG
        if (
            has_natural_language
            and IntentType.SEMANTIC_SEARCH.value in classification_result.intent_types
        ):
            return True

        return False

    def _detect_failure_patterns(
        self, question_lower: str, intents: List[str]
    ) -> Optional[str]:
        """Detect potential failure pattern (C/D/E/F) from question text and intents for evaluator hints."""
        for pattern, indicators in self.FAILURE_PATTERN_PATTERNS.items():
            for indicator in indicators:
                if re.search(indicator, question_lower, re.IGNORECASE):
                    return pattern.value

        # Additional heuristics
        if IntentType.COUNT.value in intents:
            return FailurePattern.PATTERN_C.value

        # Pattern D: Only trigger on explicit ordering requirements, not just temporal filtering
        # RECENT intent always requires ordering (most recent, latest, newest)
        # TEMPORAL alone can be filtering (published in 2024) which doesn't need ORDER BY
        # Pattern D is specifically for missing ORDER BY in ordering queries
        if IntentType.RECENT.value in intents:
            return FailurePattern.PATTERN_D.value

        # TEMPORAL with explicit ordering keywords should trigger pattern_d
        # But TEMPORAL alone (just filtering by year) should NOT
        if IntentType.TEMPORAL.value in intents:
            # Check for explicit ordering indicators that would require ORDER BY
            has_ordering_keywords = bool(
                re.search(
                    r"\b(most recent|latest|newest|oldest|earliest|first|last|top|bottom)\b",
                    question_lower,
                    re.IGNORECASE,
                )
            )
            if has_ordering_keywords:
                return FailurePattern.PATTERN_D.value

        if IntentType.STATISTICAL.value in intents:
            return FailurePattern.PATTERN_F.value

        return None
