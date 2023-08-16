from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from VGGnet.net import vgg

if __name__=="__main__":
    # ---------------------------------#
    # Cuda       是否使用Cuda
    #            没有GPU可以设置成False
    # ---------------------------------#
    Cuda = False
    # ---------------------------------#
    # 分类类型
    # ---------------------------------#
    num_classes = ['cat', 'dog']
    # ---------------------------------#
    # 'vgg16' and  'vgg19'
    # ---------------------------------#
    Netmode = 'vgg16'
    # ------------------------------------------------------------------------------#
    # detection_mode用于指定测试的模式:
    #
    # 'predict'           表示单张图片预测
    # 'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹
    # ------------------------------------------------------------------------------#
    detection_mode = "dir_predict"
    # -------------------------------------------------------#
    #   model_path指向log文件夹下的权值文件
    #   训练好后log文件夹下存在多个权值文件，选择验证集损失较低的即可。
    # -------------------------------------------------------#
    model_path = r"log\loss_2023_08_16_13_52_51\DogandCat6.pth"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在 detection_mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    device = torch.device("cuda" if torch.cuda.is_available() and Cuda else "cpu")

    model = vgg(mode=Netmode,num_classes=len(num_classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    def predict_single_image(image_path):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()

        predicted_label = num_classes[predicted_class]
        predicted_prob = probabilities[0][predicted_class].item()
        print("Output tensor:", output)
        print("Probabilities tensor:", probabilities)
        print(f"Predicted class: {predicted_label}, Probability: {predicted_prob:.2f}")
        plt.imshow(Image.open(image_path))
        plt.title(f"Predicted class: {predicted_label}, Probability: {predicted_prob:.2f}")
        plt.axis('off')
        plt.show()


    def predict_images_in_directory(origin_path, save_path):
        import os
        os.makedirs(save_path, exist_ok=True)

        image_files = [f for f in os.listdir(origin_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        for image_file in image_files:
            image_path = os.path.join(origin_path, image_file)
            result_image_path = os.path.join(save_path, image_file)

            image = Image.open(image_path)
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                model.eval()
                output = model(image)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities).item()

            predicted_label = num_classes[predicted_class]
            predicted_prob = probabilities[0][predicted_class].item()

            print("Predicted class:", predicted_label)
            print("Predicted probability:", predicted_prob)

            plt.imshow(Image.open(image_path))
            plt.title(f"Predicted class: {predicted_label}, Probability: {predicted_prob:.2f}")
            plt.axis('off')
            plt.savefig(result_image_path)
            # plt.show()

        print("Prediction and saving complete.")

    if detection_mode == "predict":
        while True:
            image_path = input('Input image filename (or "exit" to quit): ')
            if image_path.lower() == "exit":
                break
            predict_single_image(image_path)
    elif detection_mode == "dir_predict":
        predict_images_in_directory(dir_origin_path, dir_save_path)
    else:
        raise ValueError("Invalid detection_mode")