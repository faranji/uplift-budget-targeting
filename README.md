Claude
Mapping segmentation into the LP stage and thesis structure
Önce depodaki mevcut plan dosyasına bakıyorum ki yol haritası eskiyle çelişmesin.

Ran 2 commands
Segmentasyonla başlamana gerek yok.

Neden gerekmiyor: Uplift modeli her müşterinin özelliklerini (yaş, cinsiyet, alım geçmişi) doğrudan kullanıp etkiyi müşteri düzeyinde tahmin ediyor. Önceden kümeleme yapmak bilgi kaybettirir. İlk iş, purchases tablosundan (45,8 milyon satır) müşteri başına özellik üretmek: son alışveriş, alışveriş sıklığı, toplam ve ortalama harcama, ürün ve mağaza çeşitliliği. Bu RFM benzeri bir “segmentasyon” sayılır ama kümeleme değil, özellik üretimi.

Segmentlerin yeri: Sadece LP aşamasında, kapasite sınırları için gerekir (örneğin “yaş grubu başına en fazla X iletişim”). Bunları kampanyadan önce bilinen, yorumlanabilir gruplarla tanımla (yaş bandı, RFM katmanı). Tahmin edilen uplift’e göre tanımlama, çünkü sınırın amacı modelin hatasına karşı koruma sağlamak.

Yol haritası (süreler kaba tahmin, odaklı çalışma günü):

Faz	Ne yapacaksın	Çıktı	Tez bölümü
1. Veri ve özellikler (2–3 gün)	uplift_train ile clients birleştir. Yaşı temizle, first_redeem_date’i kullanma. purchases’i DuckDB veya chunksize ile parça parça oku, müşteri başına özellik üret.	Özellik tablosu	2.1, 2.2
2. Denge kontrolü ve EDA (1 gün)	Tüm özelliklerle denge ve AUC kontrolü. Yaş ve cinsiyete göre etki tablosu.	Denge tablosu	2.1, 3.1
3. Çerçeve ve baseline (1–2 gün)	Sabit test seti ayır, geri kalanda iç içe tekrarlı CV. Two-model baseline ve yanıt-olasılığı baseline’ı, Qini eğrisi.	İlk Qini eğrisi	2.6
4. Modeller (2–3 gün)	S-learner, sınıf dönüşümü, X-learner, ağaç/orman tabanlı model. Aynı hat, aynı bölmeler.	Karşılaştırma tablosu	2.3, 3.1
5. Ayar (2 gün)	Optuna ile iç döngüde ayar.	Seçilen model	2.6
6. Karar katmanı (2 gün)	Segmentleri tanımla, değer ve maliyet varsayımlarını kur, LP’yi yaz, baseline politikalarla karşılaştır.	LP kodu	2.4, 3.2
7. Politika değerlendirme (1–2 gün)	Test setinde IPW ile her politikanın artımlı alışverişini tahmin et, bootstrap aralığı.	Bütçeye göre politika eğrileri	2.6, 3.2
8. Parametrik duyarlılık (2–3 gün)	Bütçe, maliyet ve değer değişince optimal kâr, seçilen küme ve gölge fiyat.	Duyarlılık şekilleri	2.5, 3.3
9. Yazım (sürekli)	Tartışma, sonuç, özet (en son), listeler, ekler.	Tam tez	3.4, 4
Önümüzdeki ilk üç iş:

Faz 1: özellik tablosunu üret. purchases’taki sütunları ilk bakışta kontrol et.
Faz 2: tüm özelliklerle dengeyi tekrar test et. Bugünkü kontrol sadece dört özelliği kapsıyordu.
Faz 3: two-model baseline ve ilk Qini eğrisi.
Yol haritası, tez bölümleriyle birlikte depodaki PLAN.md §5'e de işlendi. Hoca takvimi ve ara teslim tarihlerini söylediğinde süreleri ona göre güncelleriz. İstersen Faz 1'in Colab not defterini hazırlayayım.

Pasted text.txt
Pasted text.txt
TXT
o halde readme'yi vr dosya yapısını değiştiricez sanırım böyle bi dosyam vardı

Betik yollarının ve eski Hillstrom dosyalarının taşınması planlanıyor
Hillstrom’dan X5'e geçtiğimiz için depodaki dosyaların hangilerinin eskidiği, hangilerinin kaldığı ve README’nin neresinin değişmesi gerektiği önce bakıyorum.

Created a file, shared 2 files, and 14 more steps
Depoyu X5'e göre yeniden düzenledim ve README’yi baştan yazdım. İkisi de zip içinde. Betikleri yeni konumlarında sentetik X5 verisiyle çalıştırdım, ikisi de çalışıyor. Gerçek veriyle denemedim, çünkü buradan indirilemiyor.

Yeni yapı

README.md, requirements.txt, .gitignore
docs/        PLAN.md, TOPLANTI_NOTU.md
notebooks/   01_x5_veri_kontrol_colab.ipynb, 02_x5_etki_analizi_colab.ipynb
scripts/     veri_seti_kontrol.py, x5_etki_analizi.py
reports/     Bitirme_Tezi_Taslak_Giris_DD050.docx
archive/hillstrom/   ilk_bakis.py, hediye_testi.py, eski ilerleme raporu, hillstrom_tab.json
data/        (git'e girmez)
Gönderdiğin ilk_bakis.py depodakiyle aynıydı, o yüzden archive/hillstrom/ altına aldım. Klasörün içine kısa bir README ekledim. Bu dosyalar artık kullanılmıyor ama referans olarak duruyor.

README’de değişenler

Veri bölümü Hillstrom yerine X5: tablolar, doğrulanan bulgular ve atama mekanizmasının belgelenmediği notu. first_redeem_date’in özellik olarak kullanılmadığı da yazıyor. Hillstrom ve Criteo’nun neden seçilmediği iki cümleyle geçiyor.
Motivasyon kısmına senin yaş bandı tablon girdi (Cochran Q = 42.7, p < 0.001).
Araştırma soruları ve LP tasarımı Giriş’teki katkılarla aynı. Tek tedavi olduğu için LP’nin değeri segment kapasite sınırlarında ve parametrik analizde. Bunu README’de “hocayla onaylanacak” diye işaretledim.
Yol haritası, docs/PLAN.md’deki 9 faza göre onay kutularıyla yazıldı. Veri seçimi, denge kontrolü ve Giriş taslağı işaretli, Faz 1'den itibaren boş.
Kaynakçaya Zhao, Goldenberg, Sun, Panagopoulos, Wager ve diğerleri eklendi. Gutierrez’in yılını tezle tutarlı olsun diye 2017 yaptım.
requirements.txt’ye Faz 1 için duckdb ve pyarrow ekledim. Bu ikisini diğer paketlerle birlikte çalıştırmadım, dosyada da öyle yazıyor.
Dikkat edeceğin yerler

Sandbox git deposu değil, o yüzden commit atmadım. Zip’i açıp içeriğini kendi GitHub klasörüne kopyalaman gerekiyor. Eski dosyaların taşındığını görmek için önce eski dosyaları silip yenilerini yapıştır, sonra git add -A ile ekle.
Tez Word dosyasını oluşturan JS dosyasını depoya koymadım. Taslağı artık Word’de elle düzenlediğin için, o dosya her çalıştığında senin düzenlemelerini ezerdi.
Colab not defterleri kendi betiklerini yazıyor, içlerindeki komutlar değişmedi. Eskisi gibi “Tümünü çalıştır” yeterli.
Sıradaki adım Faz 1. İstersen purchases tablosundan müşteri başına RFM ve çeşitlilik özelliklerini üreten Colab not defterini hazırlayabilirim.

Readme
Document
·MD
Uplift budget targeting
ZIP
Compacted conversation
Compacted conversation





Readme · MD
# Uplift Modeling + Linear Programming for Budget-Constrained Customer Targeting
 
> Whom should a marketing campaign contact when the budget is limited? This project estimates each customer's *incremental* response to a campaign (uplift modeling), turns those estimates into a budget-constrained selection with linear programming, and studies how the selection changes with the budget, the contact cost and the value of a purchase (parametric sensitivity analysis).
 
**Working title:** An Uplift Modeling and Linear Programming-Based Approach for Optimal Customer Selection under Budget Constraints in Marketing Campaigns
 
**Status (23 October 2026):** the data set is chosen and checked, the Introduction of the thesis is drafted, and feature engineering is next. There are no modeling results yet. See the [roadmap](#8-roadmap).
 
Senior graduation project, Mathematical Engineering, Yıldız Technical University (YTÜ).
 
---
 
## 1. Motivation
 
A response model answers "who is likely to buy?". A campaign decision needs a different answer: "who buys *because of* the campaign?". Some customers buy anyway, some never buy, and some may react negatively to being contacted. Contacting the first two groups wastes budget and contacting the third group destroys value.
 
Uplift modeling estimates the individual treatment effect
 
```
uplift(x) = P(Y = 1 | X = x, treated) - P(Y = 1 | X = x, control)
```
 
from data in which a control group received no communication. Given these estimates, a firm still has to decide whom to contact under a limited budget, and the best decision depends on parameters (budget, contact cost, value of a purchase) that are uncertain in practice. This project connects the estimation step and the decision step.
 
The effect is also not the same for everyone. In the X5 data used here, 60.3% of the customers who received no communication bought anyway, and the communication raised the purchase rate by 3.3 percentage points on average. The effect grows with age:
 
| Age band | n | Control % | Treated % | Effect (pp) | 95% CI |
|---|---|---|---|---|---|
| 18–29 | 29,223 | 58.10 | 59.42 | +1.32 | [0.19, 2.44] |
| 30–39 | 44,682 | 59.28 | 61.64 | +2.37 | [1.46, 3.27] |
| 40–49 | 38,468 | 59.45 | 62.06 | +2.60 | [1.63, 3.58] |
| 50–59 | 35,610 | 60.51 | 64.67 | +4.16 | [3.16, 5.17] |
| 60+ | 48,775 | 63.59 | 68.89 | +5.31 | [4.47, 6.14] |
 
The effects differ across age bands (Cochran's Q = 42.7, p < 0.001). Older customers also buy more without any communication, so ranking by purchase probability and ranking by uplift are not the same thing. (Output of `notebooks/02_x5_etki_analizi_colab.ipynb`; 3,281 customers with a missing or implausible age are excluded from this table.)
 
## 2. Research questions
 
- **RQ1 (modeling).** How do the main uplift approaches (two-model, S-learner, class transformation, X-learner and a tree-based model) compare when hyperparameters are tuned on an uplift metric and the models are evaluated with repeated, nested, stratified cross-validation instead of a single split?
- **RQ2 (decision).** How much does an LP-based selection under a budget and segment capacity limits improve the expected incremental profit over simple policies (random selection, contacting everyone, ranking by response probability, top-k by uplift)?
- **RQ3 (sensitivity).** How do the optimal profit, the selected customers and the shadow price of the budget change as the budget, the contact cost and the value of a purchase vary?
## 3. Positioning against related work
 
| Work | What it does | What this project adds |
|---|---|---|
| Gutierrez & Gérardy (2017) | Reviews causal inference and uplift modeling; compares approaches on simulated data with a single split and no costs. | Real campaign data, tuned models, repeated nested cross-validation, and a decision layer with costs and a budget. |
| Verbeke et al. (2012) | Profit-driven evaluation of churn models (maximum profit criterion), targeting by churn probability. | Targets by *incremental* effect, and optimizes the whole selected set under a budget instead of choosing a single cut-off. |
| Zhao & Harinen (2019); Goldenberg et al. (2020); Sun et al. (2024) | Allocation of treatments under costs, ROI or budget limits (several treatments, knapsack and ILP formulations). | An open, reproducible pipeline on a public data set, and a sensitivity analysis of the decision with respect to the budget, the cost and the value of a purchase. |
| Temelcan Ergenecosar et al. (2026) | Simplex-based algorithms for parametric linear programs (uncertain costs, left- or right-hand sides). | Applies parametric LP to the customer selection problem. |
| scikit-uplift | Uplift models, Qini/AUUC metrics, a maximum profit uplift measure. | No new metric is claimed. The library is a toolbox. The contribution is the comparison design, the decision layer and the sensitivity analysis. |
 
The full literature review is in the Introduction of the thesis draft (`reports/`).
 
### Relation to my earlier project
 
An earlier project, [causal-crm-uplift-modeling](https://github.com/faranji/causal-crm-uplift-modeling), built a three-arm two-model pipeline on the Hillstrom e-mail data (single 70/30 split, Qini curves, `visit` as the outcome). This thesis changes both the data and the question: from "can customers be ranked by uplift?" to "which customers should a limited budget buy, and how stable is that decision?". My first exploration on Hillstrom is kept in [`archive/hillstrom/`](archive/hillstrom/) for reference only.
 
## 4. Data
 
**Main data set: X5 RetailHero** (X5 Retail Group, uplift modeling contest, 2019).
 
| Table | Content |
|---|---|
| `uplift_train.csv.gz` | 200,039 clients: `client_id`, `treatment_flg`, `target` (purchase after the communication) |
| `clients.csv.gz` | 400,162 clients: `first_issue_date`, `first_redeem_date`, `age`, `gender` |
| `purchases` | 45,786,568 rows of purchase history before the communication (used in Phase 1) |
 
Facts checked on the data (`notebooks/01_x5_veri_kontrol_colab.ipynb`):
 
- Treatment share 0.500. Purchase rate 60.33% in control and 63.65% in treatment.
- Covariates are balanced between the groups (all standardized mean differences below 0.01).
- A classifier that predicts the treatment from the covariates reaches an AUC of 0.518. The signal comes from `first_redeem_date` alone, so **that column is not used as a feature**.
- Age is valid only between 18 and 90. Missing or implausible ages are set to missing.
**Assignment mechanism.** The documentation of the data does not state the channel, the purchase window or how clients were assigned to the groups. Randomization is therefore verified empirically and reported as a limitation, not assumed.
 
**Why not other data sets.** Hillstrom (64,000 customers) has no demographic variables. Criteo (about 14 million rows) has 12 hashed features, 85% treated customers and a conversion rate of 0.29%. Both are cited in the thesis as less suitable. Only one data set is used.
 
**Costs and values are not in the data.** The contact cost and the value of a purchase are explicit assumptions and are varied in the sensitivity analysis.
 
The raw files are downloaded to `data/` (git-ignored) and are not committed.
 
## 5. Methods
 
**Uplift models** (single treatment, binary purchase outcome)
 
1. Two-model approach (separate models for treated and control, T-learner)
2. S-learner (one model with the treatment indicator as a feature)
3. Class transformation (Jaśkowski & Jaroszewicz)
4. X-learner (Künzel et al.), implemented by hand
5. Tree-based model (uplift tree or causal forest; implementation to be decided)
Base learners are gradient boosting and logistic regression. Hyperparameters are tuned with Optuna on an uplift metric.
 
**Evaluation**
 
- Qini curve and coefficient, uplift curve and AUUC, uplift@k
- A fixed hold-out test set, and repeated nested stratified cross-validation (stratified on treatment and outcome) on the rest. Tuning happens only in the inner loop.
- Policy value: incremental purchases and profit of each policy on the hold-out set, estimated by inverse-probability weighting (IPW) with a bootstrap interval.
**Decision layer (linear programming)**
 
For customer `i` with estimated uplift `τ_i`, value of a purchase `v`, contact cost `c` and selection variable `x_i ∈ [0, 1]`:
 
```
maximize    Σ_i (v·τ_i − c) · x_i
subject to  Σ_i c · x_i      ≤ B        (budget)
            Σ_{i ∈ S_g} x_i  ≤ N_g      (capacity of segment g)
            0 ≤ x_i ≤ 1
```
 
With a single treatment and only a budget constraint, the problem reduces to sorting customers by uplift and cutting at the budget. The value of the LP therefore lies in the **segment capacity limits** and in the parametric analysis. Segments are business segments known before the campaign (age band, RFM tier), not segments defined by predicted uplift, so that the limits protect against errors of the model.
 
> **Open point:** this LP design (segment capacities with a single treatment) is to be confirmed with the advisor.
 
**Parametric sensitivity analysis**
 
The budget `B` is a right-hand-side parameter. The optimal value of a maximization problem is a concave, piecewise linear function of `B`, and its slope is the shadow price of the budget. The first version solves the LP on a grid of budgets, costs and values with SciPy/HiGHS and reports the breakpoints and the shadow prices. A simplex-based parametric algorithm (Temelcan Ergenecosar et al.) is a possible extension, not a commitment.
 
 
## 6. Assumptions and limitations
 
- Contact costs and the value of a purchase are assumptions. No data set provides them, so the conclusions hold for the tested ranges.
- The assignment mechanism of X5 is not documented. Balance between the groups is checked empirically, which cannot rule out all differences in unobserved variables.
- The data cover a single campaign and a single treatment. Results may not carry over to other channels, periods or treatments.
- The LP treats estimated uplifts as known. In reality they are noisy, and the optimal selection inherits that noise.
- Customers are assumed not to influence each other (no interference between units).
## 7. References
 
- Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. *Journal of Marketing Research*, 55(1), 80–98.
- Bertsimas, D., & Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization*. Athena Scientific.
- Bokelmann, B., & Lessmann, S. (2022). Improving uplift model evaluation on RCT data. arXiv:2210.02152.
- Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling. *Big Data*, 6(1), 13–41.
- Diemert, E., Betlei, A., et al. (2021). A large scale benchmark for individual treatment effect prediction and uplift modeling. arXiv:2111.10106.
- Goldenberg, D., Albert, J., et al. (2020). Free lunch! Retrospective uplift modeling for dynamic promotions recommendation within ROI constraints. *RecSys 2020*, 486–491.
- Gutierrez, P., & Gérardy, J.-Y. (2017). Causal inference and uplift modeling: A review of the literature. *JMLR: Workshop and Conference Proceedings*, 67, 1–13.
- Hillstrom, K. (2008). The MineThatData e-mail analytics and data mining challenge. MineThatData blog.
- Hitsch, G. J., Misra, S., & Zhang, W. (2024). Heterogeneous treatment effects and optimal targeting policy evaluation. *Quantitative Marketing and Economics*, 22(2), 115–168.
- Jaśkowski, M., & Jaroszewicz, S. (2012). Uplift modeling for clinical trial data. *ICML Workshop on Clinical Data Analysis*.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156–4165.
- Panagopoulos, G., Malitesta, D., et al. (2024). Uplift modeling under limited supervision. arXiv:2403.19289.
- Radcliffe, N. J. (2007). Using control groups to target on predicted lift. *Direct Marketing Analytics Journal*.
- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688–701.
- Rzepakowski, P., & Jaroszewicz, S. (2012). Decision trees for uplift modeling with single and multiple treatments. *Knowledge and Information Systems*, 32(2), 303–327.
- Sun, Z., Yang, H., et al. (2024). End-to-end cost-effective incentive recommendation under budget constraint with uplift modeling. arXiv:2408.11623.
- Temelcan Ergenecosar, G., Kocken, H. G., Albayrak, I., & Sivri, M. (2026). Solution to the single parametric linear programming problems via simplex-based algorithms: handling the uncertainties in costs, left or right-hand sides. *Croatian Operational Research Review*, 17(1), 29–40.
- Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.
- X5 Retail Group (2019). X5 RetailHero uplift modeling contest data. RetailHero Hackathon, Open Data Science.
- Zhao, Z., & Harinen, T. (2019). Uplift modeling for multiple treatments with cost optimization. *IEEE DSAA 2019*. arXiv:1908.05372.
The thesis draft cites 29 sources in total. This list contains the ones that shape the code and the design.
 
## 8. Author and advisor
 
- Author: Simanur Hattapoğlu
- Advisor: Prof. Dr. İnci Albayrak
