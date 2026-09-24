"""Central prompt registry and builders for every LLM workflow in BenchMind.

Provider-specific API configuration stays in the service modules. All text sent
to a model as system instructions or user prompt content is composed here.
"""

from __future__ import annotations

from typing import Any, Mapping


GEMINI_RESEARCH_INSTRUCTIONS = (
    "BenchMind için kaynaklara dayalı competitive benchmark üreten titiz bir analistsin. "
    "Son yanıtı verilen JSON sözleşmesine eksiksiz uydur. Şirket adı, URL ve yerleşik teknik "
    "terimler dışında kullanıcıya gösterilen bütün içerikleri doğal ve profesyonel Türkçe yaz; "
    "İngilizce kanıt metinlerini aynen kopyalama."
)

GROUNDED_CHAT_INSTRUCTIONS = (
    "Sen BenchMind AI'sın. Yalnızca verilen araştırma dataset'i ve retrieval ile getirilen kaynak chunk'larından "
    "yanıt ver. Dış bilgi kullanma ve web'de arama yapma. Destekleyici kaynak ID'lerini [S01] örneğindeki gibi "
    "metin içinde belirt. Doğrulanmış bilgileri, müşteri görüşlerini ve AI çıkarımlarını birbirinden ayır. "
    "Dataset yanıtı desteklemiyorsa tam olarak 'Mevcut araştırma dataset'inde yeterli kanıt yok.' cümlesiyle "
    "yanıt ver. Ardından Live Research'ü çalıştırmaya yönelik tek bir kısa öneri ekleyebilirsin. Yanıtı kısa, "
    "karar odaklı ve doğal Türkçe yaz; yerleşik teknik terimleri English bırak."
)

FEEDBACK_ANALYST_INSTRUCTIONS = (
    "You are a privacy-safe product feedback analyst. Answer only from the "
    "retrieved anonymized evidence. If the evidence is weak or incomplete, say so."
)


def build_research_instructions(max_searches: int) -> str:
    return (
        "Titiz bir competitive product research analistisin. Yanıtlamadan önce web'de arama yap. "
        f"Toplam en fazla {max_searches} ayrı web araması yap. "
        "Kaynakları şu sırayla tercih et: resmî ürün sitesi; resmî fiyatlandırma veya dokümantasyon; "
        "güvenilir review platform'ları; topluluk yorumları. Satıcı marketing içeriğini asla müşteri görüşü sayma. "
        "Doğrudan kaynağa dayanan iddiaları Verified fact, toplu review temalarını Customer opinion ve "
        "kendi sentezini AI inference olarak etiketle. Her bilgi ve görüş bulgusu geçerli kaynak ID'lerine referans "
        "versin. Verilen resmî URL, şirket kimliği için öncelikli bilgidir. Şirket adıyla URL çelişirse URL'deki "
        "gerçek markayı kullan ve benchmark'ı o marka için oluştur. Fiyat, özellik, rating, URL veya citation "
        "uydurma. Kaynaklar çelişiyorsa iki iddiayı da koru ve conflict ekle. Güncel bilgileri kullan. Metin "
        "alanlarını doğal ve profesyonel Türkçe yaz; benchmark, dataset, API, AI, transcription, integration, "
        "automation, workflow, CRM ve benzeri yerleşik teknik terimleri English bırak. Puanlar bilgi değil, "
        "analitik değerlendirmedir; toplanan kanıtlardan tutarlı biçimde hesapla."
    )


def build_research_prompt(
    companies: list[dict[str, str]],
    criteria: Mapping[str, tuple[str, float]],
    research_date: str,
) -> str:
    target = companies[0]["name"]
    company_lines = "\n".join(
        f"- {'HEDEF' if index == 0 else 'RAKİP'}: {company['name']}"
        + (f" | beklenen resmî URL: {company['website']}" if company["website"] else "")
        for index, company in enumerate(companies)
    )
    criteria_lines = "\n".join(
        f"- {key}: {label} ({weight:.0%})" for key, (label, weight) in criteria.items()
    )
    return f"""Aşağıdaki şirketler için sektör-bağımsız, güncel bir competitive benchmark oluştur.

Araştırma tarihi: {research_date}
Ürünler:
{company_lines}

Puanlama kriterleri:
{criteria_lines}

Şirketlerin ortak sektörünü ve iş modelini önce kaynaklardan belirle; onları AI toplantı asistanı veya SaaS
olarak varsayma. Her şirket için özellikle şu üç profil alanını doldur:
- pricing: güncel fiyatlandırma, üyelik veya gelir modeli;
- free_plan: ücretsiz hesap/katman/deneme varsa kapsamı; yoksa veya kavram iş modeline uygulanamıyorsa bunu
  gerekçesiyle açıkça belirt;
- target_customer: resmî konumlandırmaya dayanan ana müşteri/kullanıcı segmentleri.
Bu alanları boş bırakma ve yalnızca "Bulunamadı" yazma; yeterli kanıt yoksa hangi bilginin doğrulanamadığını açıkla.
Sektöre uygulanabilen ürün/hizmet yeteneklerini, platform ve kanal desteğini, teknoloji/automation özelliklerini,
ekip/admin/güvenlik yeteneklerini ve tekrarlanan müşteri review temalarını araştır. Sabit puanlama kriterlerinden
sektöre doğrudan uygulanmayanları en yakın iş karşılığıyla yorumla ve bu yorumu bulgularda şeffafça belirt.
SWOT'u ve önerileri özellikle {target} için analiz et. Canonical ve doğrudan erişilebilir URL'ler kullan.
Her şirket için en az bir resmî kaynak ekle ve erişilebildiğinde review platform kanıtı kullan.
Tüm puanları 0 ile 10 arasında tut."""


def build_research_json_contract(criteria: Mapping[str, Any]) -> str:
    score_keys = ", ".join(f'"{key}": 0' for key in criteria)
    return f"""
Yalnızca geçerli bir JSON object döndür; Markdown veya açıklama ekleme. Alan sözleşmesi:
- research_date, industry, target_company, executive_summary: string
- companies: 2-4 object; her object name, website, pricing, free_plan, target_customer,
  key_strength, key_limitation ve scores alanlarını içersin. scores tam olarak {{{score_keys}}} anahtarlarını kullansın.
- findings: 6-12 kısa object; id, company, category, statement, evidence_type, source_ids alanlarını içersin.
- conflicts: en fazla 3 object; her object topic, description, source_ids alanlarını içersin.
- swot: strengths, weaknesses, opportunities, threats listelerini içeren object. Her madde statement,
  evidence_type ve source_ids alanlarını içersin; her listede 1-2 kısa madde olsun.
- recommendations: 3-4 kısa object; title, priority, statement, evidence_type, source_ids alanlarını içersin.
- sources: 4-12 object; id, title, publisher, source_type, url alanlarını içersin.
evidence_type yalnızca Verified fact, Customer opinion veya AI inference; priority yalnızca High, Medium veya Low olsun.
Şirket adı, URL ve yerleşik teknik terimler dışında kullanıcıya gösterilen tüm metin alanlarını doğal Türkçe yaz.
İngilizce araştırma notlarını aynen kopyalama; anlamını koruyarak Türkçeleştir.
Metin alanlarını özlü tut; her statement en fazla iki kısa cümle olsun.
""".strip()


def build_gemini_search_prompt(research_prompt: str) -> str:
    return (
        f"{research_prompt}\n\nGoogle Search aracını mutlaka kullan. Bulguları kaynak başlıkları ve URL'leriyle "
        "birlikte ayrıntılı araştırma notları olarak döndür; henüz JSON üretme."
    )


def build_gemini_structuring_prompt(evidence: str, json_contract: str) -> str:
    return (
        "Aşağıdaki Google Search ile toplanmış kanıt paketini verilen sözleşmeye dönüştür. "
        "Yalnızca paketteki bilgileri ve URL'leri kullan; yeni bilgi veya kaynak uydurma.\n\n"
        f"{evidence}\n\n{json_contract}"
    )


def build_grounded_chat_prompt(
    question: str,
    research_context: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> str:
    retrieval_context = ""
    if retrieved_chunks:
        retrieval_blocks = []
        for chunk in retrieved_chunks:
            source_id = str(chunk.get("source_id", "KAYNAK"))
            retrieval_blocks.append(
                f"[{source_id}] chunk={chunk.get('chunk_id', '')} "
                f"başlık={chunk.get('title', '')} url={chunk.get('url', '')}\n"
                f"{chunk.get('text', '')}"
            )
        retrieval_context = (
            "\n\nRETRIEVAL İLE GETİRİLEN KAYNAK CHUNK'LARI:\n"
            + "\n\n".join(retrieval_blocks)
        )
    return (
        f"MEVCUT ARAŞTIRMA DATASET'İ:\n{research_context}"
        f"{retrieval_context}\n\nKULLANICI SORUSU:\n{question}"
    )


def build_feedback_prompt(
    dataset_label: str,
    question: str,
    retrieved_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    context_lines = []
    for index, row in enumerate(retrieved_rows, start=1):
        context_lines.append(
            f"[Chunk {index}] sentiment={row['sentiment_label']} pii={row['pii_entities'] or 'none'}\n"
            f"{row['text']}"
        )
    user_prompt = (
        f"Dataset: {dataset_label}\n"
        f"Question: {question}\n\n"
        "Retrieved anonymized feedback:\n"
        f"{chr(10).join(context_lines)}\n\n"
        "Return a concise answer with: summary, risks, and next action."
    )
    return FEEDBACK_ANALYST_INSTRUCTIONS, user_prompt
