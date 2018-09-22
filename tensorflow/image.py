import matplotlib.pyplot as plt
import tensorflow as tf



def interpolation(img_data):
    p.add_subplot(2, 2, 1)
    plt.title('Bilinear interpolation')
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    plt.imshow(resized.eval())

    p.add_subplot(2, 2, 2)
    plt.title('Nearest neighbor interpolation')
    resized = tf.image.resize_images(img_data, [300, 300], method=1)
    plt.imshow(resized.eval())

    p.add_subplot(2, 2, 3)
    plt.title('Bicubic interpolation')
    resized = tf.image.resize_images(img_data, [300, 300], method=2)
    plt.imshow(resized.eval())

    p.add_subplot(2, 2, 4)
    plt.title('Area interpolation')
    resized = tf.image.resize_images(img_data, [300, 300], method=3)
    plt.imshow(resized.eval())

def crop_or_pad(img_data):
    p.add_subplot(2, 1, 1)
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 250, 250)
    plt.imshow(croped.eval())

    p.add_subplot(2, 1, 2)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 750, 750)
    plt.imshow(padded.eval())



def draw(img_data):
    img_data=tf.image.resize_images(img_data,[250,250],method=1)
    batched=tf.expand_dims(
        tf.image.convert_image_dtype(img_data,tf.float32),0
    )
    boxes=tf.constant([[
        [0.2,0.2,0.75,0.75],
        [0.35,0.47,0.5,0.56]
    ]])
    result=tf.image.draw_bounding_boxes(batched,boxes)
    plt.imshow(result[0].eval())
    plt.show()


with tf.Session() as sess:
    image_raw_data = tf.gfile.FastGFile(r'C:\Users\Natsu\Desktop\test0.jpg', 'rb').read()
    img_data=tf.image.decode_jpeg(image_raw_data)
    #   print(img_data.eval())
    p=plt.figure()
    #   plt.show()
    img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    draw(img_data)








