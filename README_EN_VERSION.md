# PixelToProperty — English version

## Čo je v tomto balíku
Kompletná anglická verzia appky, pripravená na nahradenie súborov v tvojom
GitHub repe (`app.py`, `digitization.py`, `engineering_properties.py`,
`true_curve.py`). `requirements.txt` a `packages.txt` sú nezmenené — dávam
ich sem len pre úplnosť balíka.

## Čo bolo zmenené
1. **Celé UI v `app.py` preložené do angličtiny** — nadpisy, tlačidlá,
   popisky, hlásenia (success/warning/error), CSV export.
2. **Nové úvodné okno** (`@st.dialog`), ktoré sa zobrazí automaticky pri
   prvom otvorení appky (raz za reláciu v prehliadači, kým používateľ
   neklikne "Get started"). Obsahuje krátky návod v 3 krokoch a kontaktný
   email: `chatpall+pixeltoproperty@gmail.com`.
3. **Kontakt navyše aj v bočnom paneli** (sidebar), aby bol viditeľný počas
   celej práce s appkou, nielen v úvodnom okne.
4. **Preložené používateľské hlásenia**, ktoré appka zobrazuje pri chybách
   alebo pri diagnostike spoľahlivosti výsledku — tieto texty pochádzajú z
   `digitization.py` (chyby pri detekcii rámu/OCR), `engineering_properties.py`
   (hlásenia o spoľahlivosti elastického fitu) a `true_curve.py` (hlásenie
   pri nedostupnom Hollomonovom fite, aj hodnota "undetermined" namiesto
   pôvodného "neurcite", ktorá sa priamo zobrazuje vo výsledkoch).
5. **Výpočtová logika je nezmenená** — menili sa len reťazce (texty), nič
   v matematike/algoritmoch. Súbory som po úprave skontroloval
   (`python -m py_compile`) — syntakticky sú v poriadku.

## Čo NEBOLO menené
- Interné vývojárske komentáre v kóde (za `#`, `"""..."""` docstringy
  vysvetľujúce logiku pre teba/budúcich vývojárov) — tie ostali po
  slovensky, keďže ich používateľ appky nikdy neuvidí. Ak by si chcel
  preložiť aj tieto, daj vedieť — je to samostatná (a väčšia) úloha.
- `ZAZNAM_PROBLEMOV.md` — dokumentácia pre teba, nie súčasť appky.

## Ako to nasadiť
1. V GitHub repe nahraď `app.py`, `digitization.py`, `engineering_properties.py`,
   `true_curve.py` týmito súbormi (rovnaké názvy, priamy prepis).
2. Push do repa — Railway automaticky spustí nový build a redeploy.
3. Over si po nasadení, že sa pri prvom otvorení appky zobrazí úvodné okno
   a že celé UI je v angličtine.

## Poznámka k testovaniu
Rovnako ako pri Dockerfile — nemám tu prístup na internet, takže som appku
nemohol reálne spustiť cez `streamlit run app.py` a vizuálne overiť úvodné
okno naživo. Syntax je overená, logika zmien je priama (len text), ale
odporúčam prvé spustenie po nasadení sledovať, či sa dialóg zobrazí a zavrie
podľa očakávania.
