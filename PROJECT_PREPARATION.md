# BenchMind AI — Proje Hazırlık Dokümanı

**Durum:** Geliştirmeye hazır  
**Hazırlık tarihi:** 9 Eylül 2026  
**Proje tipi:** Freelance portföy projesi  
**MVP geliştirme hedefi:** 1–2 gün

## 1. Proje özeti

BenchMind AI, şirket ve ürünler hakkında internette dağınık halde bulunan rekabet verilerini kaynaklı, karşılaştırılabilir ve aksiyona dönüştürülebilir ürün içgörülerine çeviren bir Competitive Product Benchmarking AI uygulamasıdır.

Uygulama, GoNative'ın sürekli pazar içgörüsü yaklaşımından esinlenir; ancak kısa geliştirme süresine uygun olarak AI toplantı asistanlarının rekabet analizi üzerine odaklanan dar kapsamlı bir MVP olacaktır.

**Ürün adı:** BenchMind AI  
**Sunum sloganı:** “Turn scattered competitor data into sourced product decisions.”

## 2. Amaç ve hedef kullanıcı

### Amaç

- Çalışan ve paylaşılabilir bir freelance portföy projesi hazırlamak.
- GenAI'ın yalnızca metin üretmek için değil, kaynaklı araştırma ve ürün karar desteği için kullanılabildiğini göstermek.
- Potansiyel müşteriye veya işverene canlı olarak sunulabilecek bir uygulama ortaya çıkarmak.

### Hedef kullanıcılar

- Product manager
- Startup founder
- Product marketing manager
- Competitive intelligence veya market research uzmanı
- Küçük SaaS ekipleri ve freelance danışmanlar

## 3. Seçilen sektör ve demo şirketleri

**Sektör:** AI meeting assistants  
**Hedef ürün:** Fathom AI Notetaker (`fathom.ai`)  
**Not:** Bu ürün, `usefathom.com` adresindeki Fathom Analytics web analitiği ürünüyle karıştırılmamalıdır.

### Varsayılan rakipler

1. Fireflies.ai
2. Otter.ai
3. MeetGeek

`tl;dv`, kullanıcı tarafından seçilebilecek alternatif rakip olarak değerlendirilecektir.

## 4. Kullanıcı girdileri

Kullanıcı aşağıdaki alanları doldurabilecektir:

- Target company — zorunlu
- Target company URL — isteğe bağlı
- Competitor 1 — zorunlu
- Competitor 1 URL — isteğe bağlı
- Competitor 2 — isteğe bağlı
- Competitor 2 URL — isteğe bağlı
- Competitor 3 — isteğe bağlı
- Competitor 3 URL — isteğe bağlı

Bir hedef şirketle birlikte en fazla üç rakip karşılaştırılacaktır. URL alanları, aynı veya benzer isimli ürünlerin doğru şekilde ayırt edilmesini sağlamak için kullanılacaktır.

## 5. Temel uygulama akışı

```text
Company names + optional URLs
             |
             v
   Load Demo / Run Live Research
             |
             v
  Evidence extraction and labeling
             |
             v
Benchmark table + competitor scores
             |
             v
 SWOT + strategic recommendations
             |
             v
  Grounded AI chat + CSV export
```

## 6. Araştırma yaklaşımı

Uygulamada hibrit veri yaklaşımı kullanılacaktır.

### Hazır demo modu

`Load Demo` butonu:

- Önceden hazırlanmış ve kontrol edilmiş Fathom rekabet verisini yükler.
- Sunum sırasında API veya internet sorunu yaşansa bile uygulamanın gösterilebilmesini sağlar.
- Gereksiz API maliyeti oluşturmaz.
- Tekrarlanabilir bir sunum deneyimi sağlar.

Demo verisi, kaynakları ve araştırma tarihiyle birlikte yerel, sürümlenebilir bir veri dosyasında tutulacaktır.

### Canlı araştırma modu

`Run Live Research` butonu:

- Girilen şirketler hakkında güncel internet araştırması yapar.
- Resmî sayfaları, fiyatlandırma bilgilerini, ürün özelliklerini ve erişilebilir kullanıcı görüşlerini inceler.
- Yeni araştırmayı mevcut oturumun benchmark verisi haline getirir.
- Araştırma tarihi ve kaynak bağlantılarını sonuçlarla birlikte saklar.

Tekrarlanan sorguların maliyetini ve bekleme süresini azaltmak için uygun süreli cache kullanılacaktır.

## 7. Kaynak ve kanıt politikası

### Kaynak önceliği

1. Resmî ürün sitesi
2. Resmî fiyatlandırma sayfası ve ürün dokümantasyonu
3. Güvenilir ürün inceleme platformları
4. Topluluk yorumları

Yalnızca herkese açık şirket, ürün, fiyatlandırma ve toplulaştırılmış kullanıcı görüşü verileri kullanılacaktır. Özel müşteri bilgileri veya kişisel veri toplanmayacaktır.

### Bulgu etiketleri

Her bulgu aşağıdaki sınıflardan biriyle etiketlenecektir:

- `Verified fact`: Resmî veya güvenilir bir kaynakla doğrulanabilen bilgi.
- `Customer opinion`: Kullanıcı incelemesi veya topluluk yorumundan elde edilen öznel görüş.
- `AI inference`: Kaynaklardaki bilgilerden model tarafından çıkarılan, doğrudan kaynakta yazmayan yorum.

AI inference etiketli içerik hiçbir zaman doğrulanmış gerçek gibi sunulmayacaktır.

### Çelişkili bilgiler

Farklı kaynaklarda çelişen fiyat veya özellik bilgisi bulunursa:

- Bilgilerden biri sessizce seçilmeyecektir.
- Kullanıcıya `Conflicting evidence` uyarısı gösterilecektir.
- İlgili kaynaklar birlikte sunulacaktır.
- Resmî ve daha güncel kaynağa öncelik verilecektir.

## 8. MVP özellikleri

### Benchmark table

Ürünler aşağıdaki başlıklarda karşılaştırılacaktır:

- Pricing and free plan
- Meeting platforms and recording
- Transcription and language support
- AI summaries and action items
- Integrations and workflow automation
- Team collaboration
- Administration, privacy and security
- Target customer
- Customer sentiment
- Key strengths and limitations
- Evidence type and sources

### Source-grounded AI chat

- Sohbet yalnızca yüklenmiş araştırma veri setini kullanacaktır.
- Cevaplarda uygun kaynak bağlantıları gösterilecektir.
- Veri setinde yeterli kanıt yoksa model şu ifadeyi kullanacaktır: `Insufficient evidence in the current research dataset.`
- Kullanıcıya gerekirse `Run Live Research` çalıştırması önerilecektir.
- Model, kanıt bulunmadığında tahmin üretmeye zorlanmayacaktır.

### SWOT analysis

Hedef şirket için aşağıdaki dört bölüm üretilecektir:

- Strengths
- Weaknesses
- Opportunities
- Threats

SWOT maddeleri mevcut benchmark ve kanıtlardan türetilecek, uygun olduğunda `AI inference` olarak işaretlenecektir.

### Competitor scoring charts

Ürünler şeffaf ve sabit bir puanlama sistemiyle değerlendirilecektir:

| Kriter | Ağırlık |
|---|---:|
| Pricing & value | %15 |
| Recording & platform support | %15 |
| Transcription & languages | %15 |
| AI summaries & action items | %20 |
| Integrations & automation | %15 |
| Team, admin & security | %10 |
| Customer sentiment | %10 |

Her kriter 10 üzerinden puanlanacaktır. Uygulama, toplam puanla birlikte puanın gerekçesini ve dayandığı kaynakları gösterecektir.

Planlanan görseller:

- Kategori karşılaştırması için radar chart
- Toplam puan karşılaştırması için horizontal bar chart

### Strategic recommendations

Hedef şirket için kanıta dayalı öneriler üretilecektir:

- Positioning opportunity
- Pricing or packaging opportunity
- Product feature gap
- Target customer opportunity
- Marketing message
- Quick win
- Longer-term action

### CSV export

- Benchmark tablosu CSV olarak indirilebilecektir.
- CSV; araştırma tarihi, bulgu türü ve kaynak bağlantılarını içerecektir.
- Ayrıntılı PDF raporu MVP kapsamına dahil değildir.

## 9. Arayüz

**Arayüz dili:** Türkçe; alana özgü teknik terimler standard English biçimleriyle korunur  
**Rapor ve AI cevap dili:** Türkçe; teknik terimler English bırakılır

Önerilen ana bölümler:

1. Hero/header ve proje değer önerisi
2. Şirket giriş paneli
3. Research kontrolleri
4. Yönetici özeti
5. Benchmark tablosu
6. Rakip puan grafikleri
7. SWOT analizi
8. Stratejik öneriler
9. BenchMind'a Sor chat'i
10. Kaynaklar ve CSV export

Her araştırmada `Son araştırma` tarihi görünür olacaktır.

## 10. Teknik mimari

### Teknoloji seçimi

- Python 3.12
- Streamlit
- OpenAI Python SDK
- OpenAI Responses API
- Pandas
- Plotly
- Yerel yapılandırma için python-dotenv

Model adı kod içine sabitlenmeyecek, ortam değişkeniyle değiştirilebilir olacaktır.

### Durum ve veri saklama

- Aktif analiz ve sohbet geçmişi için Streamlit session state kullanılacaktır.
- Kontrol edilmiş demo verisi proje içindeki veri dosyalarında tutulacaktır.
- İlk sürümde kullanıcı hesabı veya kalıcı veritabanı bulunmayacaktır.
- Community Cloud'un geçici dosya sistemine kalıcı depolama amacıyla güvenilmeyecektir.

## 11. API ve güvenlik

- Uygulamanın çalışma zamanındaki GenAI ve web araştırması OpenAI API anahtarı ve API bakiyesiyle çalışacaktır.
- Codex aboneliği uygulamanın geliştirme sürecinde kullanılacak; Streamlit uygulamasının API tüketiminin yerine geçmeyecektir.
- API anahtarı kaynak koduna veya GitHub'a eklenmeyecektir.
- Yerelde `.env`, Community Cloud'da Streamlit Secrets kullanılacaktır.
- `.env` ve yerel secrets dosyaları `.gitignore` içinde tutulacaktır.
- Canlı araştırmada şirket/rakip sayısı ve istek boyutu sınırlandırılacaktır.
- Hazır demo modu API çağrısı yapmadan çalışacaktır.
- Public uygulamada kontrolsüz API tüketimini azaltacak kullanım sınırları uygulanacaktır.

## 12. Geliştirme ve dağıtım

### Geliştirme

- Uygulama yerel Windows ortamında geliştirilecek ve test edilecektir.
- Yerel Python sürümü Community Cloud ile uyumlu olacak şekilde Python 3.12 olarak hedeflenecektir.
- Uygulama repository kökünden çalıştırılacaktır.

### Teslimat ve portföy

- Kod GitHub repository üzerinden yönetilecektir.
- Son sürüm Streamlit Community Cloud'a deploy edilecektir.
- Önce private test yapılacak, ardından portföy için public hale getirilecektir.
- Canlı sunumdan önce Cloud uygulaması açılarak uyku durumundan çıkarılacaktır.
- Yerel sürüm sunum sırasında yedek olarak hazır tutulacaktır.

## 13. Yerel bilgisayar uygunluğu

Kontrol edilen geliştirme bilgisayarı:

| Bileşen | Değer | Sonuç |
|---|---:|---|
| CPU | Intel Core i5-13400F | Çok yeterli |
| Çekirdek | 10 çekirdek / 16 mantıksal işlemci | Çok yeterli |
| RAM | 15,8 GB | Yeterli |
| Boş disk | 211,3 GB | Fazlasıyla yeterli |
| İşletim sistemi | Windows 11 Pro 64-bit | Uygun |
| GPU | Gerekli değil | OpenAI modelleri API üzerinden çalışacak |

Donanım geliştirme için yeterlidir. Başlıca dış bağımlılıklar internet bağlantısı, API bakiyesi, API hız limitleri ve araştırma yanıt süresidir.

## 14. Mevcut repository için teknik not

Çalışma alanında şu anda `Privacy-Safe Feedback Analyzer` adlı önceki bir Streamlit prototipine ait dosyalar bulunmaktadır. Mevcut `requirements.txt` dosyasında bu prototip için kullanılan ağır veya yeni MVP'de gereksiz olabilecek paketler vardır:

- anthropic
- chromadb
- sentence-transformers
- transformers
- torch

BenchMind AI modelleri yerelde çalıştırmayacağı için bu bağımlılıkların yeni MVP'de kullanılması planlanmamaktadır. Geliştirmeye başlarken mevcut kodun kontrollü biçimde dönüştürülmesi ve `requirements.txt` dosyasının sadeleştirilmesi gerekir. Kullanıcıya ait mevcut dosyalar incelenmeden topluca silinmeyecektir.

## 15. MVP dışında bırakılanlar

- Ayrıntılı PDF raporu
- Kullanıcı hesabı ve kimlik doğrulama sistemi
- Kalıcı veritabanı
- Otomatik günlük/haftalık rakip takibi
- E-posta uyarıları
- Digital twin özelliği
- Kurumsal rol ve yetki yönetimi
- Özel veya kişisel müşteri verisi toplama

Bu özellikler daha sonraki ürün sürümlerinde değerlendirilebilir.

## 16. Başarı kriterleri

MVP aşağıdaki şartlar sağlandığında tamamlanmış kabul edilecektir:

- Hazır Fathom demosu internet veya API çağrısı olmadan açılabiliyor.
- Kullanıcı hedef şirket ve en fazla üç rakip girebiliyor.
- İsteğe bağlı URL alanları çalışıyor.
- Canlı araştırma kaynak bağlantılarıyla sonuç üretiyor.
- Bulgular doğru kanıt etiketiyle gösteriliyor.
- Çelişkili bilgiler görünür biçimde uyarılıyor.
- Benchmark tablosu ve iki puanlama grafiği oluşturuluyor.
- SWOT ve stratejik öneriler araştırma verisinden üretiliyor.
- Sohbet yalnızca mevcut kanıtlara dayanıyor.
- Yetersiz kanıtta açık uyarı veriliyor.
- Benchmark CSV olarak indirilebiliyor.
- API anahtarı kaynak kodunda bulunmuyor.
- Uygulama hem yerelde hem Community Cloud'da çalışıyor.

## 17. Uygulama öncesi kontrol listesi

- [x] Portföy amacı belirlendi.
- [x] Sektör belirlendi.
- [x] Hedef ürün ve varsayılan rakipler belirlendi.
- [x] MVP özellikleri belirlendi.
- [x] Hibrit veri yaklaşımı belirlendi.
- [x] Kaynak ve kanıt politikası belirlendi.
- [x] Puanlama kriterleri belirlendi.
- [x] Teknoloji seçimi yapıldı.
- [x] Yerel geliştirme ve Cloud teslimat yaklaşımı belirlendi.
- [x] Bilgisayar uygunluğu doğrulandı.
- [ ] Python 3.12 geliştirme ortamı hazırlanacak.
- [x] Bağımlılıklar sadeleştirildi.
- [x] Proje dosya mimarisi oluşturuldu.
- [x] Demo araştırma verisi hazırlandı.
- [x] Uygulama geliştirildi ve yerel testleri tamamlandı.
- [ ] Community Cloud deploy testi yapılacak.
