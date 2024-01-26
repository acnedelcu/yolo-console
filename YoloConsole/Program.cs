using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace YoloConsole
{
    public class Program
    {
        const int DimBatchSize = 1;
        const int DimNumberOfChannels = 3;
        const int ImageSizeX = 640;
        const int ImageSizeY = 640;
        const string ModelInputName = "images";
        const string ModelOutputName = "output0";

        byte[] _model;
        byte[] _sampleImage;
        List<string> _labels;
        InferenceSession _session;

        public async Task ProcessInputData()
        {
            var assembly = GetType().Assembly;

            // Get labels
            using var labelsStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.imagenet_classes.txt");
            using var reader = new StreamReader(labelsStream);

            string text = await reader.ReadToEndAsync();
            _labels = text.Split(new string[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries).ToList();

            // Get model and create session
            using var modelStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.yolov8m-cls.onnx");
            using var modelMemoryStream = new MemoryStream();

            modelStream.CopyTo(modelMemoryStream);
            _model = modelMemoryStream.ToArray();
            _session = new InferenceSession(_model);

            // Get sample image
            using var sampleImageStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.SampleImages.bus.jpg");
            using var sampleImageMemoryStream = new MemoryStream();

            sampleImageStream.CopyTo(sampleImageMemoryStream);
            _sampleImage = sampleImageMemoryStream.ToArray();
        }

        public Task<string> GetClassificationAsync()
        {
            var input = GetImageTensor(_sampleImage);

            using var results = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(ModelInputName, input)
            });

            var output = results.FirstOrDefault(i => i.Name == ModelOutputName);
            var scores = output.AsTensor<float>().ToList();
            //double s = 0;
            //for (int i = 0; i < scores.Count; i++)
            //{
            //    s += scores[i];
            //}
            var highestScore = scores.Max();
            var highestScoreIndex = scores.IndexOf(highestScore);
            var label = _labels.ElementAt(highestScoreIndex);
            return Task.FromResult(label);
        }

        private byte[] ResizeImage(byte[] image)
        {
            using var sourceBitmap = SKBitmap.Decode(image);
            byte[] pixels = sourceBitmap.Bytes;

            ////Resize
            if (sourceBitmap.Width != ImageSizeX || sourceBitmap.Height != ImageSizeY)
            {
                float ratio = (float)Math.Min(ImageSizeX, ImageSizeY) / Math.Min(sourceBitmap.Width, sourceBitmap.Height);

                using SKBitmap scaledBitmap = sourceBitmap.Resize(new SKImageInfo(
                    (int)(ratio * sourceBitmap.Width),
                    (int)(ratio * sourceBitmap.Height)),
                    SKFilterQuality.Medium);

                var horizontalCrop = scaledBitmap.Width - ImageSizeX;
                var verticalCrop = scaledBitmap.Height - ImageSizeY;
                var leftOffset = horizontalCrop == 0 ? 0 : horizontalCrop / 2;
                var topOffset = verticalCrop == 0 ? 0 : verticalCrop / 2;

                var cropRect = SKRectI.Create(
                    new SKPointI(leftOffset, topOffset),
                    new SKSizeI(ImageSizeX, ImageSizeY));

                using SKImage currentImage = SKImage.FromBitmap(scaledBitmap);
                //SaveImage(currentImage.Encode().ToArray(), "currentImage.jpg");
                using SKImage croppedImage = currentImage.Subset(cropRect);
                //SaveImage(croppedImage.Encode().ToArray(), "croppedImage.jpg");
                using SKBitmap croppedBitmap = SKBitmap.FromImage(croppedImage);

                pixels = croppedBitmap.Bytes;
            }
            return pixels;
        }

        private DenseTensor<float> GetImageTensor(byte[] image)
        {
            using (var sourceBitmap = SKBitmap.Decode(image))
            {
                byte[] pixels = ResizeImage(image);
                //normalize
                var bytesPerPixel = sourceBitmap.BytesPerPixel;
                var rowLength = ImageSizeX * bytesPerPixel;
                var channelLength = ImageSizeX * ImageSizeY;
                var channelData = new float[channelLength * 3];
                var channelDataIndex = 0;

                for (int y = 0; y < ImageSizeY; y++)
                {
                    var rowOffset = y * rowLength;
                    for (int x = 0, columnOffset = 0; x < ImageSizeX; x++, columnOffset += bytesPerPixel)
                    {
                        var pixelOffset = rowOffset + columnOffset;

                        var pixelR = pixels[pixelOffset];
                        var pixelG = pixels[pixelOffset + 1];
                        var pixelB = pixels[pixelOffset + 2];

                        var rChannelIndex = channelDataIndex;
                        var gChannelIndex = channelDataIndex + channelLength;
                        var bChannelIndex = channelDataIndex + (channelLength * 2);

                        channelData[rChannelIndex] = (pixelR / 255f - 0.485f) / 0.229f;
                        channelData[gChannelIndex] = (pixelG / 255f - 0.456f) / 0.224f;
                        channelData[bChannelIndex] = (pixelB / 255f - 0.406f) / 0.225f;

                        channelDataIndex++;
                    }
                }

                // create tensor
                var input = new DenseTensor<float>(channelData, new[]
                {
                DimBatchSize,
                DimNumberOfChannels,
                ImageSizeX,
                ImageSizeY
            });

                return input;
            }
        }

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

        public static async Task Main()
        {
            Program p = new();
            await p.ProcessInputData();
            string result = await p.GetClassificationAsync();
            Console.WriteLine($"The result is: {result}");
        }
    }
}
