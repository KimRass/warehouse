from PIL import Image
import pytesseract
pytesseract.image_to_string()

# provide the cropped area with text
def GetOCR(tempFilepath, languages='eng'):
    img = Image.open(tempFilepath)
    #img= img.convert('L')
    # filters can be applied optionally for reading the proper text from the image
    img.load()
    # -psm 5 will assume the text allinged vertically 
    text = pytesseract.image_to_string(
        image=img, lang=languages, config='-psm 6'
    )
    print(text)


if __name__ == "__main__":
    GetOCR(
        tempFilepath="/Users/jongbeom.kim/Downloads/1.png",
        languages="jap"
    )