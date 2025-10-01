# Générateur PDF à partir d’un Markdown (audit)
import markdown2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

md_file = "RELEASE_AUDIT_2025.md"
pdf_file = "RELEASE_AUDIT_2025.pdf"

with open(md_file, "r", encoding="utf-8") as f:
    md_text = f.read()

html = markdown2.markdown(md_text)

# Simple conversion : on ignore le HTML, on imprime le texte brut
lines = md_text.splitlines()

c = canvas.Canvas(pdf_file, pagesize=A4)
width, height = A4
x, y = 2*cm, height - 2*cm
c.setFont("Helvetica", 12)
for line in lines:
    if y < 2*cm:
        c.showPage()
        y = height - 2*cm
        c.setFont("Helvetica", 12)
    c.drawString(x, y, line)
    y -= 16
c.save()
print(f"PDF généré : {pdf_file}")
