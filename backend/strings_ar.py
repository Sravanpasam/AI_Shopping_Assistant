"""
strings_ar.py
-------------
Pre-translated static strings for the chatbot (English + Arabic).

Static replies (greetings, clarifying questions, "not found" messages, error
messages and comparison-table labels) are translated once by hand and stored
here. They are NOT sent through the online translator. Only dynamic text
(product titles, review snippets, policy sentences, generated summaries) goes
through ``translation.translate_to_arabic``.

Usage:
    from strings_ar import get_string
    get_string("greeting", "ar")
    get_string("search_found", "ar", count=3, plural="s")
"""

# Table row labels produced by comparison.py, in English -> Arabic.
TABLE_LABELS = {
    "Feature": "الخاصية",
    "Price": "السعر",
    "Brand / Store": "العلامة التجارية / المتجر",
    "Rating": "التقييم",
    "Category": "الفئة",
    "Key Features": "أبرز الميزات",
}

_STRINGS = {
    "greeting": {
        "en": (
            "Hello! 👋 I'm **ShopBot**, your AI-powered shopping assistant.\n\n"
            "I can help you:\n"
            "• 🔍 Find products using natural language\n"
            "• ⚖️ Compare products side-by-side\n"
            "• 📋 Answer questions about warranty, returns & delivery\n\n"
            "What are you looking for today?"
        ),
        "ar": (
            "مرحبًا! 👋 أنا **ShopBot**، مساعد التسوق الذكي الخاص بك.\n\n"
            "يمكنني مساعدتك في:\n"
            "• 🔍 العثور على المنتجات باللغة الطبيعية\n"
            "• ⚖️ مقارنة المنتجات جنبًا إلى جنب\n"
            "• 📋 الإجابة عن أسئلة الضمان والإرجاع والتوصيل\n\n"
            "عمّا تبحث اليوم؟"
        ),
    },
    "search_found": {
        "en": "I found **{count}** product{plural} for you:",
        "ar": "لقد وجدت **{count}** من المنتجات المناسبة لك:",
    },
    "search_filtered": {
        "en": "Showing results {filter}:",
        "ar": "عرض النتائج {filter}:",
    },
    "search_fallback": {
        "en": "No products found {filter}. Here are the {count} cheapest relevant products:",
        "ar": "لم يتم العثور على منتجات {filter}. إليك {count} من أرخص المنتجات ذات الصلة:",
    },
    "no_match": {
        "en": (
            "I couldn't find a match for that. 🛍️ Our store sells **beauty and personal care** "
            "products. Try rephrasing — for example: *'moisturizer for dry skin'* or "
            "*'hair dryer under $50'*."
        ),
        "ar": (
            "لم أتمكن من العثور على نتيجة مطابقة. 🛍️ متجرنا يبيع منتجات **الجمال والعناية الشخصية**. "
            "جرّب إعادة الصياغة — مثال: *'مرطب للبشرة الجافة'* أو *'مجفف شعر بأقل من 50 دولارًا'*."
        ),
    },
    "compare_intro": {
        "en": "Here's a detailed comparison:",
        "ar": "إليك مقارنة تفصيلية:",
    },
    "compare_ask": {
        "en": (
            "Which products would you like me to compare? Tell me their "
            "names (e.g. *'compare shampoo and conditioner'*), or search "
            "for items first and then say *'compare these'*."
        ),
        "ar": (
            "ما المنتجات التي تريد أن أقارنها؟ أخبرني بأسمائها "
            "(مثال: *'قارن بين الشامبو والبلسم'*)، أو ابحث عن العناصر أولاً "
            "ثم قل *'قارن هذه'*."
        ),
    },
    "clarify": {
        "en": (
            "I'm not fully sure what you need. 🤔 Could you tell me a bit more?\n"
            "For example: are you looking for a **product**, asking about a "
            "**policy** (returns, warranty, delivery), or wanting to "
            "**compare** items?"
        ),
        "ar": (
            "لست متأكدًا تمامًا مما تحتاجه. 🤔 هل يمكنك توضيح المزيد؟\n"
            "مثال: هل تبحث عن **منتج**، أم تسأل عن **سياسة** (الإرجاع، الضمان، التوصيل)، "
            "أم تريد **مقارنة** عناصر؟"
        ),
    },
    "policy_need_warranty": {
        "en": "Please search for a product first, then ask about its warranty.",
        "ar": "يُرجى البحث عن منتج أولاً، ثم اسأل عن ضمانه.",
    },
    "policy_need_cod": {
        "en": "Please search for a product first, then ask about delivery options.",
        "ar": "يُرجى البحث عن منتج أولاً، ثم اسأل عن خيارات التوصيل.",
    },
    "policy_need_return": {
        "en": "Please search for a product first, then ask about its return policy.",
        "ar": "يُرجى البحث عن منتج أولاً، ثم اسأل عن سياسة الإرجاع الخاصة به.",
    },
    "error_message_too_long": {
        "en": "Message is too long (max {max} characters).",
        "ar": "الرسالة طويلة جدًا (الحد الأقصى {max} حرفًا).",
    },
    "error_server": {
        "en": "Sorry, something went wrong on my end. Please try again.",
        "ar": "عذرًا، حدث خطأ من جهتي. يُرجى المحاولة مرة أخرى.",
    },
    "error_compare_asins": {
        "en": "Provide at least 2 product ASINs to compare.",
        "ar": "قدّم رمزَي منتج (ASIN) على الأقل للمقارنة.",
    },
    "translation_unavailable": {
        "en": "Sorry, the translation service is temporarily unavailable. Please try again shortly.",
        "ar": "عذرًا، خدمة الترجمة غير متاحة مؤقتًا. يُرجى المحاولة مرة أخرى بعد قليل.",
    },
    "reply_not_translated": {
        "en": "\n\n_(Response translation is temporarily unavailable; showing the English text.)_",
        "ar": "\n\n_(تعذّر ترجمة الرد حاليًا؛ هذا هو النص بالإنجليزية.)_",
    },
    "escalation_handoff": {
        "en": (
            "I'm connecting you with a human agent who can help further. "
            "Please hold on a moment. 🙋"
        ),
        "ar": (
            "سأوصلك بأحد موظفي خدمة العملاء لمساعدتك أكثر. "
            "يُرجى الانتظار قليلًا. 🙋"
        ),
    },
    "greeting": {
        "en": (
            "Hello! I'm ShopBot. I can find products, compare them side by side, "
            "or answer questions about returns, warranty, shipping, payment and "
            "orders. What would you like to do?"
        ),
        "ar": (
            "مرحبًا! أنا شوب بوت. يمكنني العثور على المنتجات ومقارنتها، "
            "أو الإجابة عن أسئلة الاسترجاع والضمان والشحن والدفع والطلبات. "
            "كيف يمكنني مساعدتك؟"
        ),
    },
    "llm_error": {
        "en": (
            "Sorry, I'm having trouble answering right now. "
            "Please try again in a moment, or ask me something else."
        ),
        "ar": (
            "عذرًا، أواجه مشكلة في الإجابة الآن. "
            "يُرجى المحاولة مرة أخرى بعد قليل، أو اسألني عن شيء آخر."
        ),
    },
    "llm_quota_exhausted": {
        "en": (
            "Sorry, I've reached my usage limit for now. "
            "Please try again later, or browse the store directly."
        ),
        "ar": (
            "عذرًا، لقد وصلت إلى حد الاستخدام في الوقت الحالي. "
            "يُرجى المحاولة لاحقًا، أو تصفّح المتجر مباشرة."
        ),
    },
}


def get_string(key: str, lang: str = "en", **kwargs) -> str:
    """
    Return the static string ``key`` in ``lang`` ('en' or 'ar'), formatted with
    ``kwargs`` when placeholders are supplied.

    Falls back to English if the key/language is unknown.
    """
    entry = _STRINGS.get(key)
    if entry is None:
        logger_msg = f"strings_ar: unknown key '{key}'"
        raise KeyError(logger_msg)
    template = entry.get(lang) or entry["en"]
    return template.format(**kwargs) if kwargs else template
