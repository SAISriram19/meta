import fitz, sys
sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open('OpenEnv Hackathon Opening Ceremony _ 25th Apr.pdf')
for i in range(len(doc)):
    page = doc[i]
    text = page.get_text().strip()
    if text:
        print(f'=== PAGE {i+1} ===')
        print(text)
        print()
