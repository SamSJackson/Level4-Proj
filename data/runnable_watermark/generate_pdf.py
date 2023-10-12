from fpdf import FPDF # fpdf2 is library name
filename = "test"
pdf_path = f"../generated_pdfs/{filename}.pdf"

model_name = "GPT-2"

pdf = FPDF()
pdf.add_page()

pdf.set_font("helvetica")
pdf.cell(txt=f"Example Output from Model: {model_name}")
pdf.write(txt="okay")
pdf.write(txt="not okay\nbob")
pdf.write(txt="hahah bitches")

pdf.output(pdf_path)

