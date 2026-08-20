# PixelToProperty — spustenie cez Docker

## Súbory v tomto balíku
- `Dockerfile` — recept na zostavenie image (Python 3.11 + tesseract-ocr + všetky Python balíky z requirements.txt)
- `docker-compose.yml` — pohodlné lokálne spustenie jedným príkazom
- `.dockerignore` — vynecháva zbytočné súbory zo zostavovania image
- `app.py`, `digitization.py`, `engineering_properties.py`, `true_curve.py`, `requirements.txt` — kód appky (nezmenený)

## Lokálne otestovanie (na tvojom počítači, kde máš nainštalovaný Docker Desktop)

```bash
docker compose up -d --build
```

Appka bude po chvíli (prvý build trvá dlhšie, sťahujú sa balíky) dostupná na:

```
http://localhost:8501
```

Zastavenie:
```bash
docker compose down
```

Logy (ak niečo nefunguje):
```bash
docker compose logs -f
```

## Nasadenie na Docker-hosting službu (Render / Railway / Fly.io)

Tieto tri súbory (`Dockerfile`, zdrojový kód appky, `requirements.txt`) stačí
nahrať do GitHub repozitára a prepojiť ho so zvolenou službou — tá si
Dockerfile nájde automaticky a appku podľa neho zostaví a spustí. Nič iné
netreba manuálne konfigurovať, port 8501 aj healthcheck sú už v Dockerfile
nastavené.

## Poznámka k testovaniu z mojej strany

Nemám v tomto prostredí prístup na internet, takže som Dockerfile nemohol
reálne spustiť a overiť build (žiadne sťahovanie balíkov cez apt/pip odtiaľto
nejde). Vychádzal som presne z `requirements.txt`, `packages.txt` a kódu
appky, čo by malo pokryť všetky závislosti — ale odporúčam prvé spustenie
urobiť u seba lokálne (`docker compose up -d --build`) a sledovať výstup,
či build prebehne bez chýb, než appku nasadíš na hosting.
