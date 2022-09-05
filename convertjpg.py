import glob
from PIL import Image
def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()
if __name__ == '__main__':
    folder = "images/"

    imageNum = (len(glob.glob("images/*")))
    for i in range(imageNum):
        image = glob.glob("images/*")[i]
        imgname = image.split("\\")
        img = Image.open("images/"+imgname[1])
        imsplit = imgname[1].split(".")
        imgbase = imsplit[0]
        imgextension = imsplit[1]


        width, height = img.size
        coorx1 = (width/2)-100
        coorx2 = (width/2) +100
        coory1 = (height/2)+100
        coory2 = (height/2) - 100
        #img = crop(image, (coorx1, coory2, coorx2, coory1), "cropped/" + (imgname[1]))
        #img = img.resize((200, 200))
        img = img.convert("RGB")
        img.save(imgbase+".jpg",quality=20,optimize=True)