# Real-World Malaria Analysis: Mali Case Study

## Dataset

**Indicator:** Malaria incidence (per 1,000 population at risk)  
**Country:** Mali (ISO: MLI)  
**Period:** 2000–2024 (25 annual data points)  
**Source:** World Health Organization (Global Health Observatory), via World Bank – processed by Our World in Data  
**URL:** <https://ourworldindata.org/grapher/incidence-of-malaria>  
**CSV download:** <https://ourworldindata.org/grapher/incidence-of-malaria.csv?v=1&csvType=full&useColumnShortNames=false>  
**License:** CC BY 4.0 (Our World in Data); original WHO data in the public domain  
**Download method:** Direct CSV download with `User-Agent` header (no API key required)

### How to re-download
```bash
curl -H "User-Agent: MalariaModel/1.0" \
  "https://ourworldindata.org/grapher/incidence-of-malaria.csv?v=1&csvType=full&useColumnShortNames=false" \
  -o full_malaria_incidence_owid.csv
```
Or use the included helper script:
```bash
python scripts/download_real_data.py
```

### Cached subset
A Mali-only subset is saved at `data/real/mali_malaria_incidence_who_2000_2024.csv`  
CSV format: `Country,Code,Year,Incidence_per_1000`

---

## Key Findings

| Metric | Value |
|--------|-------|
| Peak incidence | **441 per 1,000** population at risk (2013) |
| Lowest incidence | **326 per 1,000** (2019) |
| Mean incidence (2000–2024) | **384 per 1,000** |
| Latest incidence | **346 per 1,000** (2024) |
| Decline from peak to latest | **−21.5%** |
| Estimated total cases (2024) | **~7.4 million** clinical episodes |
| Estimated total cases (2013) | **~9.4 million** clinical episodes |
| Model calibrated R0 | **27.5** (Ross–MacDonald, a = 0.096) |
| Model equilibrium prevalence | **21.9%** |
| Model EIR (annual) | **4** infectious bites/person/year |

---

## Interpretation

1. **High and persistent burden.** Mali's WHO-reported malaria incidence has remained above 300 per 1,000 population at risk throughout 2000–2024, reflecting the intense, holoendemic *Plasmodium falciparum* transmission characteristic of the West African Sahel. An estimated 7.4 million clinical episodes occurred in 2024, making malaria the leading cause of morbidity and a major driver of under-five mortality.

2. **Modest but real progress.** The 21.5% decline from the 2013 peak (441/1,000) to 2024 (346/1,000) is attributable to the scaled-up deployment of insecticide-treated nets (ITNs), indoor residual spraying (IRS), seasonal malaria chemoprevention (SMC) for children, and improved case management under the Global Fund and PMI partnership era (post-2005). However, progress has plateaued since 2019, suggesting diminishing returns from current intervention packages.

3. **R0 well above elimination threshold.** The Ross–MacDonald basic reproduction number R0 derived from model parameters calibrated to Mali's average incidence is **27.5** (using a = 0.096, m = 6, b = c = 0.5, r = 1/200 d, mu = 0.1). With more conservative assumptions about the effective infectious period (r = 1/50 d), R0 = 68. Both estimates are far above the elimination threshold (R0 = 1), confirming that Mali requires sustained high-coverage interventions to maintain control, let alone pursue elimination.

4. **Seasonal transmission dynamics.** The SEIR-SEI model reproduces the characteristic Sahel seasonal epidemic curve: a sharp peak of clinical cases during and immediately after the June–October rainy season, driven by the explosion of *Anopheles gambiae* breeding sites, followed by a deep dry-season trough. This seasonal pattern directly informs the timing of SMC rounds (June–September) and seasonal IRS campaigns.

5. **Vector control as the primary lever.** R0 sensitivity analysis shows that the effective biting rate (a) and mosquito mortality (mu) have the strongest impact on transmission. A 30% reduction in effective biting lowers R0 by approximately 50%, explaining why ITN distribution and IRS have been the main drivers of Mali's observed incidence decline.

---

## Output Figures

| File | Description |
|------|-------------|
| `output/05_mali_incidence_trend.png` | WHO-reported malaria incidence in Mali, 2000–2024, with peak and latest annotations |
| `output/05_implied_R0_trend.png` | Implied R0 per year derived from WHO incidence via Ross–MacDonald equilibrium, overlaid with incidence trend |
| `output/05_model_vs_data.png` | SEIR-SEI model output: (a) monthly clinical incidence with seasonal pattern vs WHO average; (b) model prevalence dynamics |
| `output/05_mali_R0_sensitivity.png` | Sensitivity of R0 to biting rate, vector:human ratio, and mosquito mortality (Ross–MacDonald formula) |

---

## Reproduce with Your Own Data

### CSV format required
```
Country,Code,Year,Incidence_per_1000
Mali,MLI,2000,394.98
Mali,MLI,2001,396.22
...
```
- `Year`: integer year
- `Incidence_per_1000`: malaria incidence per 1,000 population at risk (numeric)

### Re-download data and run analysis
```bash
# Download fresh data
python scripts/download_real_data.py

# Run the full real-world analysis
python scripts/05_real_malaria_analysis.py
```

### Using a different country
Edit `scripts/05_real_malaria_analysis.py` and change the country code filter from `"MLI"` to your ISO 3166-1 alpha-3 code (e.g. `"BFA"` for Burkina Faso, `"GHA"` for Ghana, `"NGA"` for Nigeria). Adjust `POP_MALI` and `AT_RISK_FRAC` accordingly.

---

## Model Parameters

The Ross–MacDonald and SEIR-SEI models use parameters calibrated to Mali:

| Parameter | Value | Description |
|-----------|-------|-------------|
| a (biting rate) | 0.096 bites/mosquito/day | Calibrated to match WHO incidence |
| b (human infectibility) | 0.5 | Probability an infectious bite infects a human |
| c (mosquito infectibility) | 0.5 | Probability a bite on an infectious human infects a mosquito |
| m (vector:human ratio) | 6.0 | Mosquitoes per person (Sahel average) |
| r (human recovery rate) | 1/200 per day | ~200-day untreated infectious period |
| mu (mosquito mortality) | 0.1 per day | ~10-day average adult mosquito lifespan |
| dur_inf (infectious period) | 200 days | Duration of patent infection |
| inc_inc (incubation) | 12 days | Human intrinsic incubation period |
| eip (sporogony) | 12 days | Extrinsic incubation period in mosquito |
| season_amp | 0.65 | Amplitude of seasonal biting forcing |
| season_phase | −2.0 rad | Phase shift peaking in July–August |

---

*Analysis by Boï Kone, MRTC Malaria Unit, Bamako, Mali*  
*Data: WHO GHO / World Bank / Our World in Data, CC BY 4.0*
