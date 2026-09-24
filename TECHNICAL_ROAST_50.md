# BenchMind AI — 50 Soruluk Technical Roast

Bu dokümanın amacı sistemi olduğundan büyük göstermek değil, mimari tercihleri ve MVP sınırlarını teknik olarak savunabilmektir. Cevapları ezberlemek yerine her cevapta şu sırayı kullan: **karar → gerekçe → sınır → sonraki adım**.

## 30 saniyelik mimari anlatım

BenchMind AI iki aşamalı, evidence-aware bir competitive benchmarking uygulaması. İlk aşamada seçilen LLM provider web search ile resmi siteleri, dokümantasyonu ve review kaynaklarını araştırıyor; strict JSON schema altında şirketleri, bulguları, kaynakları, çelişkileri, SWOT'u ve yedi kategorideki ham puanları üretiyor. Uygulama ham puanları sabit ağırlıklı formülle yeniden hesaplıyor. İkinci aşamada kaynak sayfaları güvenli ve sınırlı bir crawler ile indiriliyor, 1.200 karakterlik overlap'li chunk'lara ayrılıyor, `all-MiniLM-L6-v2` ile yerelde embed edilip cosine space kullanan kalıcı Chroma indeksine yazılıyor. Chat, canlı web'e çıkmadan mevcut structured dataset ve semantik olarak getirilen ilk beş chunk üzerinden cevap veriyor.

## Bir dakikalık veri akışı

```text
Kullanıcı girdisi: hedef + en fazla 3 rakip + opsiyonel URL
        ↓
Live Research: OpenAI / Anthropic / Gemini / Claude Code + bounded web search
        ↓
Strict JSON contract: companies, scores, findings, conflicts, SWOT,
recommendations, sources
        ↓
Normalization + şirket kimliği doğrulama + deterministic weighted score
        ↓
Streamlit görünümü + CSV + research cache
        ↓ (background)
Public, same-domain bounded crawl → chunking → local embeddings → Chroma
        ↓
Soru → top-5 semantic retrieval → dataset + chunks → grounded answer + source IDs
```

## Ezberlenecek metrik formülü

Toplam puan:

```text
Overall = 0.15·Pricing
        + 0.15·Platform
        + 0.15·Transcription/Languages
        + 0.20·AI Summaries
        + 0.15·Integrations
        + 0.10·Team/Security
        + 0.10·Customer Sentiment
```

Her kategori 0–10 aralığında clamp edilir; eksik veya sayıya çevrilemeyen değer 0 kabul edilir; sonuç iki ondalığa yuvarlanır. Ağırlıkların toplamı 1'dir.

---

## A. Metrik ve değerlendirme soruları

### 1. “Bu puanlar bilimsel mi, yoksa LLM'in uydurduğu sayılar mı?”

**Cevap:** Toplam puan deterministic ve denetlenebilir; ancak kategori bazındaki ham 0–10 puanlar modelin kaynaklardan yaptığı analitik değerlendirmedir. Bu yüzden ürünü akademik ölçüm aracı olarak değil, kaynaklı decision-support MVP'si olarak konumluyorum. Production sürümünde her kategori için açık rubrik, birden çok değerlendirici ve calibration set gerekir.

**Takip gelirse:** “Formül objektif, ham skor tamamen objektif değil” demek en doğru savunmadır.

### 2. “Neden AI summaries yüzde 20, diğerleri yüzde 15 veya 10?”

**Cevap:** Demo sektörü AI meeting assistants olduğu için temel değer önerisinin toplantıyı aksiyona dönüştürme olduğunu varsaydım; bu nedenle summaries/action items en yüksek ağırlığa sahip. Ağırlıklar evrensel gerçek değil, ürün stratejisi varsayımıdır. Kurumsal security alıcısı için profil değiştirilmelidir.

### 3. “Ağırlıkları değiştirince lider değişebilir mi?”

**Cevap:** Evet, bu bir bug değil, multi-criteria decision analysis'ın doğasıdır. Sonuç kullanıcı önceliklerine duyarlıdır. Şu an MVP sabit ağırlık kullanıyor; doğru sonraki adım ağırlık profilleri ve sensitivity analysis eklemektir.

### 4. “Score normalization nasıl yapılıyor?”

**Cevap:** Kategoriler zaten ortak 0–10 ölçeğinde bekleniyor. Kod değeri 0 ile 10 arasında clamp ediyor, sonra ağırlıklı toplam alıyor. Rakip grubuna göre min-max normalization yapılmıyor; dolayısıyla yeni rakip eklemek mevcut ürünün puanını matematiksel olarak değiştirmiyor.

### 5. “Eksik veri neden sıfır? Bu ürünü haksız yere cezalandırmaz mı?”

**Cevap:** Evet, eksikliği sıfır saymak muhafazakâr ama sert bir varsayım. Teknik olarak şema bütün skorları zorunlu kıldığı için normal akışta eksik skor beklenmiyor; fallback yalnızca bozuk/veri dışı durumda devreye giriyor. Daha olgun tasarım `unknown` ile gerçek sıfırı ayırmalı ve coverage-adjusted score göstermelidir.

### 6. “Puanların birbirleriyle tutarlı üretildiğini nasıl garanti ediyorsun?”

**Cevap:** Şu an garanti etmiyorum; prompt aynı kriterleri ve kaynak politikasını uyguluyor, schema da alanları ve 0–10 sınırını zorluyor. Fakat cross-run ve cross-model tutarlılığı için regression dataset, rubrik tabanlı scoring ve evaluator ölçümleri henüz yok.

### 7. “Modeli değiştirince skorlar değişirse benchmark'ın anlamı kalır mı?”

**Cevap:** Aynı snapshot ve model sürümüyle demo tekrarlanabilir; live research ise zamana, kaynağa ve modele duyarlıdır. Bu yüzden research date ve provider/model bilgisi izlenebilir olmalıdır. Production'da model version pinning ve benchmark drift testi eklerdim.

### 8. “Customer sentiment'ı nasıl ölçüyorsun?”

**Cevap:** Bu MVP'de bağımsız bir sentiment classifier ya da istatistiksel örneklem yok. Model, erişebildiği güvenilir review platformlarındaki tekrar eden temaları sentezleyip `Customer opinion` olarak etiketliyor. Bu, pazarın temsilî sentiment skoru değil; erişilebilir review sinyalidir.

### 9. “Review sayısı, recency veya source quality ağırlığı var mı?”

**Cevap:** Hayır. Kaynak öncelik sırası prompt düzeyinde var; fakat review adedi, tarih, güvenilirlik ve örneklem büyüklüğü formülde nicel ağırlık olarak kullanılmıyor. Bu yüzden sentiment metriği directional kabul edilmeli.

### 10. “Confidence score neden yok?”

**Cevap:** Güven sayısını sırf model üretti diye göstermek sahte kesinlik yaratabilirdi. Mevcut tasarım bunun yerine evidence type, source link ve conflict görünürlüğü sağlıyor. Anlamlı confidence için source coverage, freshness, agreement ve evaluator calibration'dan hesaplanan ayrı bir metrik gerekir.

### 11. “Radar chart neden yanıltıcı olmasın?”

**Cevap:** Radar chart kategori desenini hızlı gösterir ama alan algısı farkları abartabilir. Bu nedenle karar metriği olarak değil keşif görseli olarak kullanılıyor; yanında exact scorecard ve overall bar chart bulunuyor.

### 12. “Overall score iki ondalık gösteriliyor; veri o kadar hassas mı?”

**Cevap:** Hayır. İki ondalık hesaplama sonucunun görünümüdür, ölçüm kesinliği iddiası değildir. Sunumda küçük farkları kesin liderlik olarak yorumlamam; yakın skorları aynı performans bandı olarak ele alırım.

### 13. “İstatistiksel anlamlılık testi yaptın mı?”

**Cevap:** Hayır; elimizde random sample ve tekrar ölçümleri olmadığı için p-value üretmek anlamsız olurdu. Bu bir evidence synthesis ve MCDA prototipi, deneysel araştırma değil.

### 14. “Ground truth nedir?”

**Cevap:** Verified fact'ler için ground truth'a en yakın referans resmi sayfa veya resmi dokümantasyondur. Customer opinion ve AI inference için tek bir ground truth yoktur. Sistem bu epistemik ayrımı etiketlerle açık eder.

### 15. “Benchmark'ın başarısını neyle ölçüyorsun?”

**Cevap:** Şu an teknik acceptance kriterleri var: schema geçerliliği, şirket eşleşmesi, skor aralığı ve hesap doğruluğu, kaynak ID bütünlüğü, export ve RAG işleyişi. Henüz ürün kalitesi için factual precision, citation correctness, retrieval recall ve decision usefulness ölçülmüyor; bunlar sonraki evaluation katmanıdır.

---

## B. Mimari ve RAG soruları

### 16. “Neden tek LLM çağrısı yerine iki aşamalı mimari?”

**Cevap:** Research ve soru-cevap farklı problemlerdir. Research geniş web coverage ve structured synthesis ister; chat ise sabit bir kanıt evreninde düşük gecikmeli, tekrarlanabilir cevap ister. Ayrıştırma maliyeti, hallucination alanını ve web'e gereksiz tekrar çıkışı azaltır.

### 17. “Structured dataset varken RAG neden gerekli?”

**Cevap:** Structured dataset karşılaştırma, skor ve özet için yeterli; fakat uzun kaynak sayfalarındaki ayrıntıları kaybeder. RAG, chat sırasında orijinal sayfa chunk'larını geri getirerek structured özetin ötesinde kanıt sağlar. İkisini birlikte prompt'a koymak hem global bağlamı hem lokal kanıtı korur.

### 18. “RAG gerçekten grounded mı?”

**Cevap:** Mimari olarak model sadece mevcut dataset ve retrieved chunk'larla sınırlandırılıyor, chat çağrısında web tool yok ve source ID isteniyor. Ancak bu, matematiksel garanti değildir; citation faithfulness evaluator henüz bulunmuyor. Doğru ifade “grounding'i teşvik eden ve denetlenebilir kılan tasarım”dır.

### 19. “Neden `all-MiniLM-L6-v2`?”

**Cevap:** Küçük, hızlı, yerelde çalışabilen ve MVP için yeterli semantic retrieval kalitesi veren bir model. API embedding maliyeti ve ek veri aktarımı oluşturmuyor. Dezavantajı multilingual/Türkçe ve domain-specific retrieval'da daha güçlü modellere göre sınırlı olabilmesi.

### 20. “İngilizce sayfalara Türkçe soru sorulursa embedding kalitesi düşmez mi?”

**Cevap:** Düşebilir; kullanılan MiniLM modeli tam multilingual optimizasyon hedeflemiyor. Structured Türkçe dataset bir miktar bağlam sağlıyor ama cross-lingual retrieval riski devam ediyor. Production tercihim multilingual-e5 veya benzeri bir modelle karşılaştırmalı retrieval eval yapmak olurdu.

### 21. “Chunk size neden 1.200 karakter, overlap neden 180?”

**Cevap:** Bu değerler bağlam bütünlüğü ile retrieval granularity arasında MVP mühendislik varsayımıdır. Kod mümkünse paragraf, cümle veya boşluk sınırından bölüyor; 180 karakter overlap sınırdaki bilginin kaybını azaltıyor. Optimal oldukları iddia edilmiyor; retrieval eval ile tune edilmeliler.

### 22. “Neden top-5 retrieval?”

**Cevap:** Varsayılan beş chunk, yeterli çeşitlilik ile prompt gürültüsü ve token maliyeti arasında pratik bir denge. Dinamik `k`, relevance threshold veya reranker şu an yok. Özellikle karşılaştırmalı sorularda per-company retrieval daha iyi olabilir.

### 23. “Similarity threshold nasıl çalışıyor?”

**Cevap:** Chroma'nın cosine distance değeri similarity'ye çevriliyor ve yalnızca similarity değeri en az 0.40 olan chunk'lar kullanılıyor. Hiçbir chunk sınırı geçmezse model çağrılmadan “yeterli retrieval kanıtı yok” yanıtı veriliyor. Türkçe sorular ve çoğunlukla İngilizce kaynaklar için multilingual embedding kullanılıyor; 0.40 başlangıç değeri gerçek BenchMind indexindeki ilgili ve alakasız örneklerle seçildi, daha geniş retrieval eval ile yeniden ayarlanmalıdır.

### 24. “Reranking kullanıyor musun?”

**Cevap:** Hayır. Chroma'daki cosine similarity sıralaması doğrudan kullanılıyor. MVP gecikmesini ve bağımlılıkları düşük tutuyor; fakat cross-encoder reranker özellikle yakın anlamlı ürün karşılaştırmalarında precision'ı artırabilir.

### 25. “Vector database neden Chroma?”

**Cevap:** Local-first, kurulumu kolay, metadata saklayabilen ve persistent collection sunan bir MVP seçimi. Community Cloud filesystem'i ephemeral olduğu için production durability sağlamaz; managed vector DB veya persistent volume gerekir.

### 26. “İndeks ne zaman invalid oluyor?”

**Cevap:** Collection adı target company, research date ve source URL'lerinin hash'inden üretiliyor; ayrıca metadata'daki embedding model eşleşmesi kontrol ediliyor. Kaynak içeriği URL değişmeden güncellenirse hash bunu algılamaz. Content hash veya TTL eklenmesi gerekir.

### 27. “Research sonucu neden hemen gösteriliyor, indeks neden background?”

**Cevap:** Benchmark için structured çıktı hazır olduğunda kullanıcı değer görmeye başlayabilir; crawl ve local embedding daha uzun sürer. Background job algılanan gecikmeyi azaltır. İndeks hazır değilse chat structured dataset ile çalışmaya devam eder.

### 28. “Crawler tüm interneti tarıyor mu?”

**Cevap:** Hayır. Seed source ve company URL'lerinden başlıyor; yalnızca aynı domain veya alt domain, varsayılan depth 2, toplam 30 document, domain başına 8 sayfa ve 30 saniye sınırı var. Bu coverage değil, kontrollü evidence enrichment yaklaşımıdır.

### 29. “SSRF'ye karşı ne yaptın?”

**Cevap:** Yalnızca HTTP(S) kabul ediliyor; DNS çözümündeki private, loopback, link-local, multicast, reserved ve unspecified IP'ler reddediliyor. Redirect'ler manuel ve en fazla altı adım izleniyor, her adımda aynı-domain ve public-host kontrolü yapılıyor. Yine de DNS rebinding gibi ileri tehditler için production hardening gerekir.

### 30. “robots.txt başarısız olursa neden crawl'a izin veriyorsun?”

**Cevap:** Mevcut politika robots dosyası alınamazsa allow ediyor; bu teknik olarak fail-open davranışıdır. Public MVP coverage'ını korur ama compliance açısından tartışmalıdır. Kurumsal sürümde fail-closed veya domain allowlist daha güvenli tercih olabilir.

### 31. “JavaScript ile render edilen sayfalarda ne oluyor?”

**Cevap:** Crawler yalnızca HTTP ile gelen HTML/PDF/plain text'i işler; browser rendering yok. Client-side içerik alınamayabilir ve failed/eksik evidence olarak kalır. Playwright gibi browser tabanlı ingestion kapsam dışı tutuldu.

### 32. “PDF, büyük dosya ve redirect sınırları nedir?”

**Cevap:** PDF metni `pypdf` ile çıkarılıyor; document başına varsayılan 3 MB sınırı, 5 saniye connection ve 12 saniye HTTP timeout'u var. Redirect zinciri altı kontrol döngüsüyle sınırlı ve domain dışına çıkamıyor.

### 33. “Crawler eşzamanlı çalışırken rate limit uyguluyor mu?”

**Cevap:** Worker sayısı varsayılan 8 ve en fazla 16 ile clamp ediliyor; fakat domain başına gecikme ya da token-bucket rate limiter yok. Sayfa sayısı ve toplam süre bounded olsa da nazik crawling için per-domain concurrency/rate limit eklenmelidir.

### 34. “Kaynak sayfası sonradan değişirse eski cevaplar ne olur?”

**Cevap:** Research snapshot ve cache zamana bağlıdır, ancak otomatik TTL veya revalidation şu an yok. Kullanıcı yeniden live research çalıştırabilir. Production'da freshness SLA, `retrieved_at`, ETag/Last-Modified ve scheduled refresh gerekir.

### 35. “Aynı URL iki farklı kaynak ID'siyle gelirse ne olur?”

**Cevap:** Crawl queue URL bazında deduplicate eder; ilk eşleşen source metadata'sı korunur. Bu işlem indirme tekrarını azaltır ama çoklu provenance ilişkisini kaybedebilir. İdeal model bir document ile birden fazla source claim arasında ilişki tablosu tutar.

---

## C. Parametre ve kural çerçevesi soruları

### 36. “Sistemin temel hard limitleri neler?”

**Cevap:** Araştırma 2–4 şirket ve varsayılan en fazla 4 web search ile sınırlandırılmıştır; output en az 2.000, varsayılan 10.000 token; chat 500 output token ve soru 1.200 karakter; RAG varsayılan 30 document, domain başına 8 sayfa, depth 2, 3 MB/document, 30 saniye, 8 worker ve top-5 retrieval kullanır.

### 37. “Kurallar kodda mı, prompt'ta mı?”

**Cevap:** İki yerde. Tip, zorunlu alan, enum, array ve skor sınırları JSON schema/kod seviyesinde; kaynak önceliği, epistemik etiketleme ve yazım davranışı prompt seviyesinde. Güvenlik ve crawl limitleri doğrudan kodla enforced ediliyor.

### 38. “Prompt injection'a karşı koruma var mı?”

**Cevap:** Kaynak metni data olarak veriliyor ve sistem talimatı dış bilgi kullanmayı yasaklıyor; chat'te web tool kapalı. Ancak source-content prompt injection için sanitization, instruction detection veya privilege-separated parsing yok. Bu nedenle tam koruma iddia etmiyorum.

### 39. “Kaynakta ‘önceki talimatları yok say’ yazarsa?”

**Cevap:** Modelin bunu içerik olarak değerlendirmesi bekleniyor ama teknik enforcement yok. Production'da kaynakları açık XML/JSON data boundary içinde sunmak, suspicious instruction filtrelemek ve answer sonrası citation/faithfulness gate çalıştırmak gerekir.

### 40. “Evidence type etiketini kim belirliyor?”

**Cevap:** LLM, prompt'taki kurala göre belirliyor; schema yalnızca üç geçerli enum değerinden birini seçmesini zorluyor. Dolayısıyla label validity garantiye yakın, label correctness garanti değil. Bunun için örneklenmiş human review ve confusion matrix gerekir.

### 41. “Verified fact gerçekten verify ediliyor mu?”

**Cevap:** Sistem iddiayı source ID'ye bağlar ve resmi kaynakları önceler, fakat her cümle için otomatik entailment testi yapmaz. Buradaki “verified” operasyonel bir evidence sınıfıdır, bağımsız üçüncü taraf doğrulaması değildir.

### 42. “Çelişkili kaynaklarda hangi bilgi kazanıyor?”

**Cevap:** Prompt iki iddiayı da korumayı ve conflict üretmeyi söyler; resmi ve güncel kaynağa öncelik verir. Ama sistem sessizce tek doğru seçmek yerine belirsizliği görünür kılmayı hedefler. Tarih metadata'sının kaynak bazında zorunlu olmaması şu an bir eksiktir.

### 43. “URL ile şirket adı çelişirse neden URL kazanıyor?”

**Cevap:** URL daha güçlü bir identity anchor; aynı isimli markaları ayırmak için kullanılıyor. Girdi temizleme URL'den şirket adı çıkarıyor, çıktı da beklenen şirketlerle hizalanıyor ve yanlış hedefli benchmark reddediliyor. Yine de typo-squatting riskine karşı domain allowlist/ownership doğrulaması yok.

### 44. “LLM invalid JSON döndürürse ne oluyor?”

**Cevap:** OpenAI tarafında strict JSON schema kullanılıyor; diğer provider akışlarında JSON contract ve parser var. Parser code fence içindeki veya metne sarılmış ilk object'i de yakalayabiliyor. Parse edilemezse araştırma hata verir; otomatik repair/retry zinciri şu an sınırlı.

### 45. “Provider'lar gerçekten eşdeğer davranıyor mu?”

**Cevap:** Hayır, ortak veri contract'ına normalize ediliyorlar ama search, structured output, reasoning ve usage özellikleri farklı. OpenAI strict schema ile tek akış; Anthropic bounded web search; Gemini search ve structuring olmak üzere iki aşama; Claude Code CLI bütçe sınırıyla çalışıyor. Provider parity test edilmesi gereken bir risk alanı.

### 46. “Cache key neleri içeriyor, neleri içermiyor?”

**Cevap:** Schema version ile normalize edilmiş şirket adı ve URL listesinin SHA-256 hash'ini içeriyor. Provider, model, tarih, prompt sürümü ve scoring rubric sürümü key'e dahil değil. Bu nedenle cache reproducibility için model/prompt/version metadata'sı eklenmeli.

### 47. “Cache ne kadar süre geçerli?”

**Cevap:** Şu an TTL yok; kayıt zamanı gösteriliyor ve kullanıcı kayıtlı sonucu veya yeniden araştırmayı seçiyor. Bu bilinçli MVP sadeliği ama güncel fiyat gibi volatile veriler için otomatik freshness politikası gerekir.

### 48. “API maliyetini nasıl kontrol ediyorsun?”

**Cevap:** Live research öncesi kullanıcı onayı var; search sayısı ve output token sınırı bounded; sonuçlar cache'leniyor; chat 500 output token ve RAG/local embeddings kullanıyor. Maliyet tahmini bazı bilinen provider/model kombinasyonları için gösterilebiliyor; gerçek fatura garantisi değildir.

### 49. “Gizlilik ve veri güvenliği iddian nedir?”

**Cevap:** Bu benchmark akışı public şirket/ürün verisi için tasarlandı; API key kodda tutulmuyor, `.env` veya Streamlit Secrets kullanılıyor, OpenAI çağrılarında `store=False` ayarlı. Kullanıcı hesabı, tenant isolation, encryption policy, audit log ve kurumsal RBAC olmadığı için hassas müşteri verisine hazır olduğunu söylemem.

### 50. “Yarın production'a al deseler ilk üç teknik yatırımın ne olur?”

**Cevap:** Birincisi evaluation: claim-level citation correctness, retrieval recall@k, score stability ve human rubric. İkincisi governance: auth/RBAC, secrets manager, tenant isolation, audit log ve retention. Üçüncüsü reliability: job queue, managed DB/vector store, retries, observability, freshness/TTL ve provider fallback. Bunlardan önce UI özelliği eklemek doğru öncelik olmaz.

---

## Sunumda gelebilecek hızlı çapraz sorular

### “En güçlü teknik kararın ne?”

Research ile chat'i ayırmak ve her ikisini aynı provenance yapısına bağlamak. Böylece araştırma web-enabled, chat ise bounded ve tekrar üretilebilir kalıyor.

### “En zayıf teknik noktan ne?”

Ham kategori skorlarının açık, insan tarafından kalibre edilmiş rubriğe dayanmaması. Şeffaf formül bu sorunu azaltıyor ama çözmüyor.

### “Bu bir agent mı, RAG mı, workflow mu?”

En doğru tanım: bounded tool-using research workflow + structured synthesis + local RAG chat. Otonom, uzun süreli plan yapan genel amaçlı agent değil.

### “Neyi bilerek kapsam dışında bıraktın?”

Kullanıcı hesabı, kalıcı cloud storage, enterprise authorization, scheduled monitoring, PDF rapor ve bilimsel accuracy benchmark. MVP'nin amacı uçtan uca evidence-aware karar desteğini göstermek.

## Söylememen gereken cümleler

- “Puanlar tamamen objektif.”
- “Hallucination'ı sıfırladım.”
- “Verified fact yüzde 100 doğrudur.”
- “RAG her zaman doğru kaynağı bulur.”
- “Uygulama production-ready ve enterprise-secure.”
- “Customer sentiment tüm müşteri kitlesini temsil eder.”
- “Her provider aynı sonucu verir.”
- “Cache her zaman günceldir.”

## Güvenli ve güçlü alternatif cümleler

- “Toplam formül deterministic; ham skorlar kaynaklara dayalı analitik değerlendirmedir.”
- “Hallucination yüzeyini kaynak sınırı, structured output ve provenance ile daralttım.”
- “MVP'nin güçlü yanı belirsizliği saklamak yerine etiketlemesi ve kaynağa bağlamasıdır.”
- “Bu sürüm karar vermez; karar verene denetlenebilir bir başlangıç noktası sağlar.”
- “Sınırları bilerek görünür tuttum; production yol haritam evaluation, governance ve reliability ile başlıyor.”

## 10 dakikalık prova yöntemi

1. Rastgele beş soru seç.
2. Her birini 30 saniyede **karar → gerekçe → sınır → sonraki adım** sırasıyla cevapla.
3. Her cevapta en fazla bir sayı kullan; sayı kullanıyorsan kaynağını koddan bil.
4. Bilmediğin noktada “Bu sürümde ölçmedim” de ve nasıl ölçeceğini söyle.
5. Son turda yalnızca 1, 6, 18, 23, 38, 41, 46, 49 ve 50. soruları çalış; en sert savunma hattı bunlardır.
