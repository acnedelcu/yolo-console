using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace YoloConsole
{
    public class Program
    {
        private MemoryStream image;
        private List<string> labels;
        private InferenceSession session;

        private const int DimBatchSize = 1;
        private const int DimNumberOfChannels = 3;
        private const int TargetWidth = 640;
        private const string ModelInputName = "images";
        private const string ModelOutputName = "output0";
        private const int TargetHeight = 640;

        /// <summary>
        /// Constructor
        /// </summary>
        public Program()
        {
            this.image = new MemoryStream();
            ReadInputImage("bus.jpg");

            this.labels = ReadLabels();
            this.session = ReadModel();
        }

        /// <summary>
        /// Reads the image to be classified and copies the stream contents to <see cref="image"/>
        /// </summary>
        /// <param name="imageName">The name of the image, with extension</param>
        /// <exception cref="ArgumentNullException">Image not found</exception>
        public void ReadInputImage(string imageName)
        {
            if (!imageName.Contains('.'))  //TODO update
            {
                throw new Exception("The name of the image does not have an extension!");
            }

            var assembly = GetType().Assembly;
            using var imageStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.SampleImages.{imageName}");
            _ = imageStream ?? throw new ArgumentNullException(nameof(imageStream));
            imageStream.CopyTo(image);
        }

        /// <summary>
        /// Read the labels from the file
        /// </summary>
        public List<string> ReadLabels()
        {
            var assembly = GetType().Assembly;
            using var labelsStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.imagenet_classes.txt");

            if (labelsStream == null || labelsStream.Length == 0)
            {
                throw new ArgumentNullException(nameof(labelsStream));
            }

            using var reader = new StreamReader(labelsStream);

            string text = reader.ReadToEnd();
            List<string> labels = text.Split(new string[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries).ToList();
            return labels;
        }

        /// <summary>
        /// Returns a new <see cref="InferenceSession"/> using the .onnx file
        /// </summary>
        public InferenceSession ReadModel()
        {
            var assembly = GetType().Assembly;
            using var modelStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.yolov8m-cls.onnx");

            if (modelStream == null || modelStream.Length == 0)
            {
                throw new ArgumentNullException(nameof(modelStream));
            }

            //copy to a MemoryStream
            using MemoryStream stream = new();
            modelStream.CopyTo(stream);
            return new InferenceSession(stream.ToArray());
        }

        /// <summary>
        /// Resizes <see cref="image"/> to the specified <see cref="TargetWidth"/> and <see cref="TargetHeight"/>
        /// </summary>
        /// <returns>The resized image</returns>
        public byte[] GetResizedImage()
        {
            byte[] sourceImage = image.ToArray();

            using SKBitmap sourceBitmap = SKBitmap.Decode(sourceImage);
            if (sourceBitmap.Width != TargetWidth || sourceBitmap.Height != TargetHeight)
            {
                float ratio = (float)Math.Min(TargetWidth, TargetHeight) / Math.Min(sourceBitmap.Width, sourceBitmap.Height);

                SKImageInfo info = new((int)(ratio * sourceBitmap.Width), (int)(ratio * sourceBitmap.Height));
                using SKBitmap scaledBitmap = sourceBitmap.Resize(info, SKFilterQuality.Medium);

                int horizontalCrop = scaledBitmap.Width - TargetWidth;
                int verticalCrop = scaledBitmap.Height - TargetHeight;
                int leftOffset = horizontalCrop == 0 ? 0 : horizontalCrop / 2;
                int topOffset = verticalCrop == 0 ? 0 : verticalCrop / 2;
                var cropRect = SKRectI.Create(new SKPointI(leftOffset, topOffset), new SKSizeI(TargetWidth, TargetHeight));

                using SKImage croppedImage = SKImage.FromBitmap(scaledBitmap).Subset(cropRect);

                //SaveImage(croppedImage.Encode().ToArray(), "resizedImage.jpg");
                return croppedImage.Encode(SKEncodedImageFormat.Jpeg, 100).ToArray();
            }
            return sourceImage;
        }

        /// <summary>
        /// Normalized the provided image
        /// </summary>
        /// <param name="image">The image to be normalized</param>
        /// <returns>The normalized, concatenated values for the RGB channels</returns>
        public float[] NormalizeImage(byte[] image)
        {
            SKBitmap bitmap = SKBitmap.Decode(image);
            var pixels = bitmap.Pixels;

            float[] rValues = new float[pixels.Length];
            float[] gValues = new float[pixels.Length];
            float[] bValues = new float[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                rValues[i] = pixels[i].Red / 255f;
                gValues[i] = pixels[i].Green / 255f;
                bValues[i] = pixels[i].Blue / 255f;
            }
            return rValues.Concat(gValues).Concat(bValues).ToArray();
        }

        /// <summary>
        /// Saves the image with the specified image name
        /// </summary>
        /// <param name="image">The image</param>
        /// <param name="imageName">The name of the image with extension</param>
        private void SaveImage(byte[] image, string imageName)
        {
            using (var ms = new MemoryStream(image))
            {
                using (var fs = new FileStream(@$"D:\{imageName}", FileMode.Create))
                {
                    ms.WriteTo(fs);
                }
            }
        }

        /// <summary>
        /// Classifies the image
        /// </summary>
        /// <param name="normalizedImageData">The normalized image to be classified</param>
        /// <returns>The classification</returns>
        public string GetClassification(float[] normalizedImageData)
        {
            var imageTensor = new DenseTensor<float>(normalizedImageData, new[]
                {
                DimBatchSize,
                DimNumberOfChannels,
                TargetWidth,
                TargetHeight
            });

            using var results = this.session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(ModelInputName, imageTensor)
            });

            var output = results.FirstOrDefault(i => i.Name == ModelOutputName);
            _ = output ?? throw new ArgumentNullException(nameof(output));

            List<float> scores = output.AsTensor<float>().ToList();
            float highestScore = scores.Max();
            int highestScoreIndex = scores.IndexOf(highestScore);
            string label = this.labels.ElementAt(highestScoreIndex);
            return label;
        }

        private static void Main(string[] args)
        {
            Program p = new();
            byte[] resizedImage = p.GetResizedImage();
            float[] normalizedImage = p.NormalizeImage(resizedImage);
            string result = p.GetClassification(normalizedImage);
            Console.WriteLine(result);
        }
    }
}