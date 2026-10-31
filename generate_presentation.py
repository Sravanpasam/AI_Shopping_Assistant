"""
generate_presentation.py
------------------------
Generates a 12-slide widescreen (16:9) PowerPoint presentation (.pptx)
for the ShopBot E-Commerce NLP & Agent System project.
Follows the university NLP course presentation requirements:
  - Dataset selection
  - Pre-processing strategy
  - Model selection and configuration
  - Training/fine-tuning process
  - Evaluation metrics and results
  - Analysis of performance and potential biases
  - Discussion of ethical and societal implications
Formats content with medium, clear typography (no tiny text), balancing
accessible executive summaries with deep technical specifications.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Theme Colors ─────────────────────────────────────────────────────────────
NAVY = RGBColor(15, 30, 54)        # Deep Navy / Header
DARK_SLATE = RGBColor(30, 41, 59)  # Body text
LIGHT_BG = RGBColor(248, 250, 252) # Soft Off-White
CARD_BG = RGBColor(255, 255, 255)  # Card Fill
BORDER_GRAY = RGBColor(226, 232, 240)
BLUE_ACCENT = RGBColor(26, 86, 219)   # Royal Blue
TEAL_ACCENT = RGBColor(13, 148, 136)  # Deep Teal
EMERALD = RGBColor(16, 149, 90)       # Success Green
CORAL = RGBColor(225, 77, 42)         # Highlight / Coral
MUTED_TEXT = RGBColor(100, 116, 139) # Subtitle text
CARD_HEADER_BG = RGBColor(238, 242, 255) # Light Indigo

FONT_HEADING = "Calibri"
FONT_BODY = "Calibri"


def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_badge="SHOPBOT NLP SYSTEM"):
        # Header banner area
        badge_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8), Inches(0.4))
        tf_badge = badge_box.text_frame
        tf_badge.word_wrap = True
        tf_badge.margin_left = tf_badge.margin_top = tf_badge.margin_bottom = tf_badge.margin_right = 0
        p_badge = tf_badge.paragraphs[0]
        p_badge.text = category_badge.upper()
        p_badge.font.name = FONT_HEADING
        p_badge.font.size = Pt(11)
        p_badge.font.bold = True
        p_badge.font.color.rgb = BLUE_ACCENT

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_bottom = tf_title.margin_right = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_HEADING
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    def add_card(slide, left, top, width, height, title, header_color=BLUE_ACCENT):
        # Container shape
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = BORDER_GRAY
        card.line.width = Pt(1.5)

        # Title bar banner inside card
        title_box = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.2), width - Inches(0.6), Inches(0.45))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_bottom = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = header_color

        # Body textbox
        body_box = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.7), width - Inches(0.6), height - Inches(0.85))
        tf_body = body_box.text_frame
        tf_body.word_wrap = True
        tf_body.margin_left = tf_body.margin_top = tf_body.margin_bottom = tf_body.margin_right = 0
        return tf_body

    def add_bullet(tf, bold_prefix, text_content, font_size=14, space_after=10):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.font.name = FONT_BODY
        p.space_after = Pt(space_after)
        
        run_bold = p.add_run()
        run_bold.text = bold_prefix + (" " if bold_prefix else "")
        run_bold.font.bold = True
        run_bold.font.size = Pt(font_size)
        run_bold.font.color.rgb = DARK_SLATE

        run_text = p.add_run()
        run_text.text = text_content
        run_text.font.bold = False
        run_text.font.size = Pt(font_size)
        run_text.font.color.rgb = DARK_SLATE

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 1: Title Slide
    # ═══════════════════════════════════════════════════════════════════════════
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, NAVY)

    # Accent decorative block
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.8), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = CORAL
    bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(3.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "ShopBot: Bilingual E-Commerce NLP System"
    p0.font.name = FONT_HEADING
    p0.font.size = Pt(36)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(255, 255, 255)
    p0.space_after = Pt(12)

    p1 = tf.add_paragraph()
    p1.text = "From Zero-Shot Semantic Retrieval to Grounded Multi-Tool Agent Orchestration"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(20)
    p1.font.color.rgb = RGBColor(199, 210, 254)
    p1.space_after = Pt(28)

    p2 = tf.add_paragraph()
    p2.text = "Master of Applied Artificial Intelligence | Natural Language Processing Capstone"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(226, 232, 240)
    p2.space_after = Pt(8)

    p3 = tf.add_paragraph()
    p3.text = "Technologies: Python • Flask • FAISS • Sentence-Transformers • Whisper ASR • Gemini / Groq"
    p3.font.name = FONT_BODY
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(148, 163, 184)

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 2: Executive Overview & Dual-System Architecture
    # ═══════════════════════════════════════════════════════════════════════════
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, LIGHT_BG)
    add_header(s2, "Executive Overview: Project Goal & System Evolution", "Project Overview")

    tf_c1 = add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "The Challenge & Core Goals", TEAL_ACCENT)
    add_bullet(tf_c1, "Real-World Problem:", "Standard keyword e-commerce search fails when users describe products colloquially ('something for dry frizzy hair') or ask policy questions.")
    add_bullet(tf_c1, "Bilingual Parity:", "Full voice & text support in both English and Arabic with automatic right-to-left layout and cultural/currency translation.")
    add_bullet(tf_c1, "Grounded Truth First:", "A strict anti-hallucination requirement: the system must never invent products, prices, or policies. Unavailable info must be stated as unavailable.")
    add_bullet(tf_c1, "Voice-First Architecture:", "OpenAI Whisper transcribes spoken audio directly in-browser, giving accessible e-commerce to all customers.")

    tf_c2 = add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Dual Architecture: System A vs System B", BLUE_ACCENT)
    add_bullet(tf_c2, "System A (Baseline - Frozen):", "A fast, local rule-and-retrieval pipeline using SentenceTransformers intent routing, FAISS dense search, and regex policy extraction. Kept frozen for scientific benchmark.")
    add_bullet(tf_c2, "System B (Active LLM Agent):", "A tool-calling shopping agent (Gemini 2.0 Flash / Groq LLaMA 3.3-70B) orchestrating 10 grounded tools in a bounded loop (≤5 rounds).")
    add_bullet(tf_c2, "Navigation Over Walls of Text:", "Instead of dumping text into chat, System B emits structured deep-link actions that dynamically drive the storefront UI (search, compare, orders).")
    add_bullet(tf_c2, "Zero-Downtime Fallback:", "Includes an automatic offline rule-based fallback net when API quotas are reached.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 3: Dataset Selection
    # ═══════════════════════════════════════════════════════════════════════════
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, LIGHT_BG)
    add_header(s3, "Dataset Selection & Structured Knowledge Repository", "Dataset Selection")

    tf_c1 = add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Non-Technical Rationale", TEAL_ACCENT)
    add_bullet(tf_c1, "Real Marketplace Benchmark:", "Used Amazon Reviews 2023 (McAuley Lab), focusing on the All_Beauty domain (~1,000 products, 9M+ reviews sampled).")
    add_bullet(tf_c1, "True Customer Voices:", "Real products with genuine customer feedback, star ratings, and unvarnished product claims rather than synthetic dummy data.")
    add_bullet(tf_c1, "Realistic Store Policies:", "Created 5 authentic store policy manuals (returns, warranty, shipping, payments/COD, order cancellations).")
    add_bullet(tf_c1, "Real-Time Transaction Mocking:", "Populated an orders database (orders.db) with tracking numbers, fulfillment statuses, and realistic delivery dates.")

    tf_c2 = add_card(s3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Deep Technical Data Schema", BLUE_ACCENT)
    add_bullet(tf_c2, "Catalog Schema (catalog.parquet):", "parent_asin, title, store, price (USD), average_rating (1-5), rating_number, category_path, features_text, description_text, n_reviews_sampled.")
    add_bullet(tf_c2, "Reviews Schema (reviews.parquet):", "parent_asin, rating, title, text, helpful_vote, verified_purchase, timestamp. Filtered to ≥5 verified reviews per item.")
    add_bullet(tf_c2, "Policy Documents (docs/*.md):", "Structured markdown files chunked by heading and paragraph for dense embedding retrieval.")
    add_bullet(tf_c2, "Order Ledger (SQLite):", "Stores order_id (ORD-XXXXX), status (processing, shipped, delivered), courier tracking, and estimated delivery dates.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 4: Pre-Processing Strategy
    # ═══════════════════════════════════════════════════════════════════════════
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, LIGHT_BG)
    add_header(s4, "Pre-Processing Strategy & Query Normalization", "Pre-Processing Strategy")

    tf_c1 = add_card(s4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Query Normalization (query_normalizer.py)", BLUE_ACCENT)
    add_bullet(tf_c1, "Shopping Filler Removal:", "Strips conversational fluff in EN & AR (e.g., 'I am looking for', 'show me', 'أبحث عن', 'من فضلك') while preserving key product nouns.")
    add_bullet(tf_c1, "Catalog Vocabulary Typo Repair:", "Uses Gestalt pattern matching (difflib) strictly against words that exist in the catalog (cutoff ≥ 0.72). Fixes 'mascarra' -> 'mascara' without mangling brand names.")
    add_bullet(tf_c1, "Zero Model Overhead:", "Entirely rule-based and vocabulary-driven; executes in under 1 millisecond on CPU.")
    add_bullet(tf_c1, "Protected Filter Tokens:", "Price keywords ('under', 'between', 'aed', '$') are protected from typo alteration so downstream parsers remain intact.")

    tf_c2 = add_card(s4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Price Parsing & Catalog Synthesis", TEAL_ACCENT)
    add_bullet(tf_c2, "Dual-Currency Parsing (price_parser.py):", "Extracts min/max bounds from natural text ('under $20', 'between 15 and 30 dollars', 'under 100 AED').")
    add_bullet(tf_c2, "Fixed Currency Conversion:", "Converts AED amounts to USD using fixed parity (3.67 AED = $1.00 USD), allowing seamless queries across both currencies.")
    add_bullet(tf_c2, "False-Positive Shield:", "Protects non-price numbers like volume, weights, and counts ('50 ml', 'pack of 2', 'spf 30') from triggering price filters.")
    add_bullet(tf_c2, "Catalog Text Fusion:", "Combines title + category_path + features + description into unified document representations for both dense and sparse indexing.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 5: Model Selection & Configuration
    # ═══════════════════════════════════════════════════════════════════════════
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, LIGHT_BG)
    add_header(s5, "Model Selection & Configuration Topology", "Model Selection")

    tf_c1 = add_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Local NLP Models (System A & Tooling)", TEAL_ACCENT)
    add_bullet(tf_c1, "Sentence-Transformers (all-MiniLM-L6-v2):", "384-dimensional dense embeddings. Ultra-fast CPU inference (~15ms), zero cloud cost, 22M parameters. Used for semantic search & policy retrieval.")
    add_bullet(tf_c1, "FAISS IndexFlatIP (Cosine Similarity):", "Computes inner products of L2-normalized vectors for instant top-50 nearest neighbor retrieval.")
    add_bullet(tf_c1, "Scikit-Learn TF-IDF Engines:", "Word N-grams (1-2) with sublinear TF scaling + Character N-grams (3-5) for exact keyword & typo matching.")
    add_bullet(tf_c1, "OpenAI Whisper (base/tiny):", "Multilingual ASR for speech-to-text. Browser audio (WebM) converted server-side to WAV via ffmpeg.")

    tf_c2 = add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Agent LLMs & Translation (System B)", BLUE_ACCENT)
    add_bullet(tf_c2, "Google Gemini 2.0 Flash:", "Primary agent orchestrator. Fast time-to-first-token, native JSON tool calling, temperature=0.2 (low randomness for factual accuracy).")
    add_bullet(tf_c2, "Groq LLaMA-3.3-70B Versatile:", "Hot-swappable OpenAI-compatible fallback provider via httpx. Activated instantly if Gemini hits daily limits.")
    add_bullet(tf_c2, "Translation Pipeline (langdetect + deep-translator):", "langdetect with fixed seed (0) for deterministic language tagging; Google Translator wrapper with thread-safe rate-limiting (3 req/sec).")
    add_bullet(tf_c2, "In-Memory LRU Cache:", "2,000-entry translation cache prevents repeated API trips for common product titles.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 6: Training, Adaptation & Retrieval Process
    # ═══════════════════════════════════════════════════════════════════════════
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, LIGHT_BG)
    add_header(s6, "Training, Adaptation & Retrieval Engineering", "Training / Adaptation")

    tf_c1 = add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Why Zero-Shot Retrieval Over Fine-Tuning?", TEAL_ACCENT)
    add_bullet(tf_c1, "The Problem with LLM Fine-Tuning:", "Fine-tuning LLMs on product catalogs causes hallucination on prices and stock changes, and requires expensive retraining whenever catalog items update.")
    add_bullet(tf_c1, "Retrieval-Augmented Architecture (RAG):", "Instead of training the LLM to memorize items, we keep data in structured stores (Parquet, FAISS, SQLite) and feed verified data dynamically via tools.")
    add_bullet(tf_c1, "Zero-Shot Dense Similarity:", "Tuned intent classification and semantic search by setting empirical similarity thresholds on the pre-trained embedding space.")
    add_bullet(tf_c1, "Fingerprinted Index Cache:", "FAISS index caches metadata (row count, SHA-256 of ASINs) and rebuilds automatically only when catalog data changes.")

    tf_c2 = add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Hybrid Reciprocal Rank Fusion (RRF)", BLUE_ACCENT)
    add_bullet(tf_c2, "The Formula:", "RRF_Score = 1.0 / (60 + DenseRank) + 0.35 / (60 + WordRank) + 0.15 / (60 + CharRank).")
    add_bullet(tf_c2, "Dense + Lexical Harmony:", "Dense vectors capture intent ('something for curly hair'); character/word TF-IDF recovers specific brand names and typos.")
    add_bullet(tf_c2, "Search Acceptance Floor (0.51):", "Top candidate must reach 0.51 similarity to be accepted as a confident match. Chosen via measure_search.py.")
    add_bullet(tf_c2, "Search Margin Pruning (0.16):", "Keeps only results within 0.16 of the top match, cleanly cutting off irrelevant low-scoring results.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 7: Tool-Grounded Agent Loop
    # ═══════════════════════════════════════════════════════════════════════════
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, LIGHT_BG)
    add_header(s7, "Tool-Grounded Agent Loop & Navigation Actions", "Agent Architecture")

    tf_c1 = add_card(s7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "10 Grounded Agent Tools (agent_tools.py)", BLUE_ACCENT)
    add_bullet(tf_c1, "Catalog Tools:", "search_products (hybrid search + price bounds), get_product_details (specs, features), search_reviews (real user feedback).")
    add_bullet(tf_c1, "Comparison Tool:", "compare_products generates deterministic side-by-side spec tables directly from catalog records.")
    add_bullet(tf_c1, "Policy & Knowledge:", "search_policies retrieves semantically matching chunks from policy documentation (floor 0.40).")
    add_bullet(tf_c1, "Order Ledger:", "check_order queries SQLite orders.db for fulfillment status, tracking IDs, and estimated arrival dates.")
    add_bullet(tf_c1, "Specialized Assist Tools:", "build_routine (regimens by goal/budget), gift_finder (targeted suggestions), best_in_category (top-rated items), escalate_to_human.")

    tf_c2 = add_card(s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Safety Boundaries & Navigation Actions", TEAL_ACCENT)
    add_bullet(tf_c2, "Strict Grounding Rules:", "Prompt explicitly forbids inventing facts. Every price, rating, date, and status must originate from a tool result in that turn.")
    add_bullet(tf_c2, "Prompt Injection Defense:", "The prompt instructs the model to treat all tool outputs (reviews, descriptions) as untrusted text, never as instructions.")
    add_bullet(tf_c2, "No Fake Actions:", "The bot cannot claim to execute refunds or cancellations. It can only report info or escalate to human support.")
    add_bullet(tf_c2, "Structured Deep Links:", "Every tool call produces an action object (type: 'navigate', page: 'search' | 'compare' | 'orders' | 'help') so the storefront UI updates in real-time.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 8: Evaluation Metrics & Benchmark Results
    # ═══════════════════════════════════════════════════════════════════════════
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, LIGHT_BG)
    add_header(s8, "Evaluation Metrics & Experimental Results", "Evaluation & Results")

    tf_c1 = add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Natural Language Search Benchmark", EMERALD)
    add_bullet(tf_c1, "Test Suite (measure_search.py):", "50 real-world queries covering 5 distinct phrasings tested against expected catalog ASINs.")
    add_bullet(tf_c1, "Bare Queries (e.g. 'JINRI hair dryer'):", "10 / 10 correct -> 100.0% Hit@3")
    add_bullet(tf_c1, "Polite Requests ('show me Bed Head wand'):", "10 / 10 correct -> 100.0% Hit@3")
    add_bullet(tf_c1, "Use Descriptions ('tousled waves barrel'):", "9 / 10 correct -> 90.0% Hit@3")
    add_bullet(tf_c1, "Typo Corrupted ('essence mascarra'):", "10 / 10 correct -> 100.0% Hit@3")
    add_bullet(tf_c1, "Nonsense / Out-of-Scope Queries:", "10 / 10 rejected -> 100.0% Rejection")
    add_bullet(tf_c1, "Overall Benchmark Accuracy:", "97.5% Hit@3 | 100.0% Nonsense Rejection", font_size=15)

    tf_c2 = add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "System A vs System B (104 Questions)", BLUE_ACCENT)
    add_bullet(tf_c2, "Benchmark Setup (run_eval.py):", "104 questions across 10 categories (products, price, policies, orders, escalation, follow-up).")
    add_bullet(tf_c2, "Product Search:", "System A: 75.0% | System B (Agent): 97.5%")
    add_bullet(tf_c2, "Price Filtering:", "System A: 100.0% | System B: 100.0%")
    add_bullet(tf_c2, "Policy Questions (Returns, Warranty, Shipping):", "System A: 0.0% (Regex failed) | System B: 95.0%+")
    add_bullet(tf_c2, "Order Tracking (SQLite):", "System A: 10.0% | System B: 100.0%")
    add_bullet(tf_c2, "Human Escalation:", "System A: 0.0% | System B: 100.0%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 9: Performance Analysis & Potential Biases
    # ═══════════════════════════════════════════════════════════════════════════
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, LIGHT_BG)
    add_header(s9, "Performance Analysis & Potential Biases", "Performance & Biases")

    tf_c1 = add_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Multilingual & Dialectal Biases", CORAL)
    add_bullet(tf_c1, "Arabic Dialect Gap:", "Standard translation engines assume Modern Standard Arabic (MSA). Colloquial dialects (Egyptian, Gulf, Levantine) can experience translation drift on slang.")
    add_bullet(tf_c1, "Bilingual Pre-Filtering Mitigation:", "Implemented dedicated Arabic character regexes and transliteration rules in query_normalizer and greetings.py before calling translation.")
    add_bullet(tf_c1, "Whisper Dialect Performance:", "Whisper 'base' model exhibits lower transcription confidence on fast Arabic speech; 'small' model provides substantial boost in Arabic WER.")
    add_bullet(tf_c1, "Bidirectional RTL Alignment:", "Frontend enforces dir='rtl' dynamically to maintain native user experience.")

    tf_c2 = add_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Catalog Skew & Review Sentiment Bias", TEAL_ACCENT)
    add_bullet(tf_c2, "Domain Vocabulary Concentration:", "Catalog is 100% beauty and personal care. Dense embeddings naturally exhibit high baseline similarity for generic cosmetic words ('smooth', 'glow').")
    add_bullet(tf_c2, "Lexical RRF Counter-Weight:", "Hybrid search balances semantic embeddings with word and character TF-IDF, preventing false matches on generic marketing claims.")
    add_bullet(tf_c2, "Review Positivity Skew:", "Sampled Amazon reviews skew heavily toward 4 and 5 stars. Pure text summaries can sound uniformly positive.")
    add_bullet(tf_c2, "Quantitative Balancing:", "System surfaces raw average ratings and total review counts directly alongside summaries to provide balanced perspective.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 10: Ethical & Societal Implications
    # ═══════════════════════════════════════════════════════════════════════════
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, LIGHT_BG)
    add_header(s10, "Ethical, Legal & Societal Implications", "Ethics & Society")

    tf_c1 = add_card(s10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Consumer Protection & Truth-in-Commerce", BLUE_ACCENT)
    add_bullet(tf_c1, "Preventing Financial Harm:", "An e-commerce bot that hallucinates a lower price, a fake refund, or a 30-day return policy creates legal liability and financial disputes.")
    add_bullet(tf_c1, "Strict Grounding Constraint:", "Every factual claim (price, warranty period, delivery charge) is programmatically validated against retrieved tool output.")
    add_bullet(tf_c1, "Clear Refusal of Unverifiable Facts:", "If information is not present in the catalog or policy documents, the assistant explicitly states it is unavailable rather than estimating.")
    add_bullet(tf_c1, "Medical Claim Defense:", "Questions regarding medical cures (e.g. 'will this cream cure my disease?') trigger safe refusal advising consultation with a healthcare professional.")

    tf_c2 = add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Authority Boundaries & Inclusion", EMERALD)
    add_bullet(tf_c2, "Execution vs Reporting Boundary:", "Autonomous AI must not possess write-access to financial or refund databases. ShopBot is read-only and explicitly directs write requests to human agents.")
    add_bullet(tf_c2, "PII Privacy Protection:", "Order tracking endpoint (/api/orders/<id>) sanitizes all personally identifiable information (customer name, address, payment card) and shows only shipment progress.")
    add_bullet(tf_c2, "Accessibility Inclusion:", "OpenAI Whisper voice input empowers visually impaired, low-literacy, and non-native English speakers to shop seamlessly.")
    add_bullet(tf_c2, "Human Escalation Protocol:", "Recognizes customer frustration and repeatedly disputed charges, instantly providing seamless handoff to human support.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 11: Production Resilience & Fallback Safety Net
    # ═══════════════════════════════════════════════════════════════════════════
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, LIGHT_BG)
    add_header(s11, "Production Resilience & Dual-Provider Architecture", "Resilience & Reliability")

    tf_c1 = add_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "Multi-Tier LLM Fallbacks & Budgeting", BLUE_ACCENT)
    add_bullet(tf_c1, "Automatic Model Rotation:", "If the primary model (Gemini 2.0 Flash) encounters a 404 (retired model) or daily quota exhaustion (429), it automatically rotates to fallback models in order.")
    add_bullet(tf_c1, "Groq Secondary Provider:", "Full cross-provider redundancy via Groq (LLaMA-3.3-70B). Switching providers requires only changing LLM_PROVIDER in .env.")
    add_bullet(tf_c1, "Daily Quota Protection:", "LLM_DAILY_BUDGET enforces a per-model ceiling (default 20 calls/day), preventing cloud API overages.")
    add_bullet(tf_c1, "Thread-Safe Rate Limiting:", "Outgoing client calls are paced by a shared sliding-window rate limiter (10 req/min).")

    tf_c2 = add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Offline Fallback Net (fallback.py)", TEAL_ACCENT)
    add_bullet(tf_c2, "Zero-Downtime Guarantee:", "When all LLM models are exhausted or network is unavailable, the system switches to fallback.py.")
    add_bullet(tf_c2, "Local Rule-Based Execution:", "Executes local FAISS search, policy extraction, and order checks using fixed templates without calling external APIs.")
    add_bullet(tf_c2, "Graceful Status Surfacing:", "GET /api/health surfaces 'quota_exhausted', updating the widget status indicator to amber without crashing.")
    add_bullet(tf_c2, "Eval Cache (LLM_CACHE):", "Stores SHA-256 hashed chat responses in data/llm_cache.json, allowing benchmark reruns without consuming LLM API calls.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 12: Summary & Technology Stack
    # ═══════════════════════════════════════════════════════════════════════════
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, LIGHT_BG)
    add_header(s12, "Summary & Technical Architecture Matrix", "Project Conclusion")

    # Add a clean 2-column technical summary
    tf_c1 = add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "End-to-End Technology Stack", NAVY)
    add_bullet(tf_c1, "Backend Application:", "Python 3.11, Flask (Host 127.0.0.1, Debug Off), Flask-CORS, Requests, Threading.")
    add_bullet(tf_c1, "Semantic Vectors & Retrieval:", "SentenceTransformers (all-MiniLM-L6-v2), FAISS (IndexFlatIP), Scikit-Learn TF-IDF.")
    add_bullet(tf_c1, "Speech & Language:", "OpenAI Whisper (ASR), langdetect, deep-translator (Google Translate API).")
    add_bullet(tf_c1, "Agent & LLMs:", "Google Gemini 2.0 Flash (google-genai SDK), Groq LLaMA 3.3-70B (httpx).")
    add_bullet(tf_c1, "Storage & Data Formats:", "Apache Parquet, PyArrow, SQLite3, JSON.")
    add_bullet(tf_c1, "Frontend Storefront:", "HTML5, CSS3, Vanilla JS (Hash Router), Shadow DOM Chat Widget.")

    tf_c2 = add_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2), "Key Takeaways & Future Horizons", EMERALD)
    add_bullet(tf_c2, "High-Accuracy Semantic Retrieval:", "97.5% Hit@3 and 100% nonsense rejection achieved through hybrid RRF dense + sparse search.")
    add_bullet(tf_c2, "Factual Reliability Guaranteed:", "Strict negative grounding constraints eliminate hallucinations on price, stock, and return rules.")
    add_bullet(tf_c2, "Bilingual Parity Achieved:", "Natural interaction in both English and Arabic with automated layout, currency, and language switching.")
    add_bullet(tf_c2, "Future Roadmap:", "On-device multilingual embeddings (e.g. multilingual-e5) to eliminate external translation dependencies entirely.")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ShopBot_Project_Presentation.pptx")
    prs.save(output_path)
    print(f"Successfully generated PowerPoint presentation at: {output_path}")
    return output_path


if __name__ == "__main__":
    create_deck()
