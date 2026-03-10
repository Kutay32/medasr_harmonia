"""RAG-powered Turkish medical report generator."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.llm_backend import get_llm
from rag.vectorstore import similarity_search

SYSTEM_PROMPT = """\
Sen deneyimli bir Türk radyoloji uzmanısın. Görevin, verilen tıbbi transkripsiyon metnini \
profesyonel bir Türkçe tıbbi rapora dönüştürmektir.

Kurallar:
1. Raporu SADECE BİR KERE yaz ve bitir. Kesinlikle aynı raporu veya formatı tekrar etme.
2. Standart Türk radyoloji rapor formatını kullan (Tetkik Türü, Endikasyon, Teknik, Bulgular, Sonuç).
3. Tıbbi terimleri doğru Türkçe karşılıklarıyla kullan.
4. Verilen bağlam belgelerindeki terminoloji ve format örneklerini referans al.
5. Eğer transkripsiyon İngilizce ise, terimleri Türkçeye çevir.
6. Profesyonel ve klinik bir dil kullan.
7. Eksik bilgileri tahmin etme; sadece verilen bilgileri kullan.
8. Rapor bittiğinde başka bir metin üretme, dur.

Bağlam belgeleri:
{context}
"""

USER_PROMPT = """\
Aşağıdaki tıbbi transkripsiyon metnini profesyonel bir Türkçe radyoloji raporuna dönüştür:

Transkripsiyon:
{transcript}

Tespit edilen dil: {language}

ÖNEMLİ: Raporu sadece BİR KERE yaz ve dur. Tekrar etme.
"""


def generate_turkish_report(
    transcript: str,
    language: str = "auto",
    llm_backend: str | None = None,
    k: int = 4,
) -> str:
    """Generate a Turkish medical report from a transcript using RAG.

    Parameters
    ----------
    transcript : str
        The raw ASR transcript (English or Turkish).
    language : str
        Detected language of the transcript ("en", "tr", or "auto").
    llm_backend : str, optional
        Override the default LLM backend.
    k : int
        Number of context documents to retrieve.

    Returns
    -------
    str
        The generated Turkish medical report.
    """
    # Retrieve relevant context from the knowledge base
    docs = similarity_search(transcript, k=k)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs) if docs else "(Bilgi tabanında belge bulunamadı.)"

    # Build the chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    llm = get_llm(llm_backend)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "context": context_text,
        "transcript": transcript,
        "language": language,
    })

    return result
