import fitz
doc = fitz.open('OpenEnv Hackathon Opening Ceremony _ 25th Apr.pdf')
print(f'Pages: {len(doc)}')
for i, page in enumerate(doc):
    imgs = page.get_images()
    if imgs:
        print(f'Page {i+1}: {len(imgs)} image(s)')
        for j, img in enumerate(imgs):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base['ext']
            path = f'_pdf_img_p{i+1}_{j+1}.{ext}'
            with open(path, 'wb') as f:
                f.write(base['image'])
            w = base.get('width', '?')
            h = base.get('height', '?')
            print(f'  saved {path} ({w}x{h})')
