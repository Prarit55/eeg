"""
Master aggregator for Modules 4 through 12.
"""
from scripts.handbook_modules_4_to_6 import (
    get_module_4_sections,
    get_module_5_sections,
    get_module_6_sections,
)
from scripts.handbook_modules_7_to_9 import (
    get_module_7_sections,
    get_module_8_sections,
    get_module_9_sections,
)
from scripts.handbook_modules_10_to_12 import (
    get_module_10_sections,
    get_module_11_sections,
    get_module_12_sections,
)

def get_modules_4_to_12():
    return [
        ("MODULE 4: DEEP NEURAL ARCHITECTURE (1D-CNN + BiLSTM HYBRID)", get_module_4_sections()),
        ("MODULE 5: DUAL KNOWLEDGE CHANNELS: VECTOR RAG & CHROMADB", get_module_5_sections()),
        ("MODULE 6: CLINICAL KNOWLEDGE GRAPHS: OPTIMUSKG & NEO4J", get_module_6_sections()),
        ("MODULE 7: QUERY UNDERSTANDING & ADAPTIVE DYNAMIC ROUTING", get_module_7_sections()),
        ("MODULE 8: MULTIMODAL EVIDENCE FUSION & CONTEXT STRUCTURING", get_module_8_sections()),
        ("MODULE 9: SELF-RAG ACTIVE REFLECTION & ITERATIVE REFINEMENT", get_module_9_sections()),
        ("MODULE 10: SOFTWARE ARCHITECTURE, REST API & WEB DASHBOARD", get_module_10_sections()),
        ("MODULE 11: COMPREHENSIVE VIVA VOCE DEFENSE MASTERCLASS", get_module_11_sections()),
        ("MODULE 12: CODE WALKTHROUGH, ABLATION BENCHMARKS & BIBLIOGRAPHY", get_module_12_sections()),
    ]
