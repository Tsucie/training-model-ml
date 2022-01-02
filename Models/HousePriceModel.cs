using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.IO;
using System.Linq;

namespace training_model_ml.Models
{
    public class HousePriceModel
    {
        // Feature Selection
        // Memilih kolom berdasarkan nama property di HouseData untuk dijadikan Feature dalam pelatihan model
        private static string[] numericFeatures = {
            "Bedrooms",
            "Bathrooms",
            "LivingArea",
            "LotArea",
            "Floors",
            "Waterfront",
            "View",
            "Condition",
            "Grade",
            "HighFeet",
            "BasementFeet"
        };

        /// <summary>
        /// Build dan melatih model yang digunakan untuk memprediksi harga rumah
        /// </summary>
        /// <param name="mLContext">ML Context Objek</param>
        /// <param name="pathData">Path ke Dataset</param>
        /// <param name="pathOutputModel">Path untuk output Dataset</param>
        public static void CreateModelUsingPipeline(MLContext mlContext, string pathData, string pathOutputModel, char separatorBy)
        {
            Console.WriteLine("Training Model Dataset ...");

            // Memuat sampel data menjadi sebuah view yang digunakan untuk melatih model
            // ML.NET mendukung berbagai macam tipe data
            var trainingDataView = mlContext.Data.LoadFromTextFile<HouseData>(
                path: pathData,
                separatorChar: separatorBy,
                hasHeader: true,
                allowQuoting: true,
                trimWhitespace: false,
                allowSparse: false
            );

            // Membuat trainer yang akan digunakan untuk melatih model
            // ML.NET mendukung metode training yang berbeda
            // Beberapa trainer mendukung fitur normalisasi otomatis dan pengaturan regulasi
            // ML.NET mendukung pemilihan beberapa algoritma yang berbeda
            var trainer = mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features"
            );

            // ML.NET menggabungkan transformasi untuk penyiapan data dan model training menjadi sebuah single pipeline,
            // kemudian di aplikasikan ke training data dan input data yang digunakan untuk membuat prediksi
            var trainingPipeline = mlContext.Transforms.Concatenate(
                outputColumnName: "Features",
                inputColumnNames: numericFeatures
            )
            .Append(trainer);

            // Melatih model
            var model = trainingPipeline.Fit(trainingDataView);

            // Save model
            using (var file = File.OpenWrite(pathOutputModel))
                mlContext.Model.Save(model, trainingDataView.Schema, file);

            Console.WriteLine("Training Complete!");
        }

        public static void CreateModelUsingCrossValidationPipeline(MLContext mlContext, string pathData, string pathTransformModel, string pathOutputModel, char separatorBy)
        {
            Console.WriteLine("Training Model Dataset ...");

            // Memuat Sampel data
            var trainingDataView = mlContext.Data.LoadFromTextFile<HouseData>(pathData, separatorChar: separatorBy, hasHeader: true);

            // Membuat Regression SDCA trainer
            var trainer = mlContext.Regression.Trainers.Sdca();

            // Feature Selection
            var dataTransformPipeline = mlContext.Transforms.Concatenate(
                outputColumnName: "Features",
                inputColumnNames: numericFeatures
            );

            Console.WriteLine("Cross Validating untuk mendapatkan ketepatan model ...");

            // Data Transform Pipeline untuk konversi HouseData training feature set
            var transformer = dataTransformPipeline.Fit(trainingDataView);
            var transformedData = transformer.Transform(trainingDataView);

            // Train dan Cross Validating model
            // Mengembalikan model dan metrik training untuk setiap fold, pada kasus ini memilih 3 folds
            var crossValidationResult = mlContext.Regression.CrossValidate(transformedData, trainer, 3);

            // Menggunakan RSquared Metrik untuk menentukan hasil fitted data
            // Semakin mendekati 1 maka semakin baik
            // Memilih semua model dan mengambil 1 yang terbaik
            ITransformer[] models =
                crossValidationResult.OrderByDescending(fold => fold.Metrics.RSquared)
                    .Select(fold => fold.Model)
                    .ToArray();

            // Model best fit
            ITransformer topModel = models[0];

            // Print Status
            Console.WriteLine($"**********************************************");
            Console.WriteLine($"* Metrics for {trainer.ToString()} Regression ");
            Console.WriteLine($"*---------------------------------------------");
            foreach (var result in crossValidationResult)
            {
                Console.WriteLine($"* R-squared: {result.Metrics.RSquared:0.###}  ");
            }
            Console.WriteLine($"**********************************************");

            // Save model file
            if (File.Exists(pathOutputModel)) File.Delete(pathOutputModel);
            if (File.Exists(pathTransformModel)) File.Delete(pathTransformModel);

            using (var file = File.OpenWrite(pathTransformModel))
                mlContext.Model.Save(transformer, trainingDataView.Schema, file);
            
            using (var file = File.OpenWrite(pathOutputModel))
                mlContext.Model.Save(topModel, transformedData.Schema, file);
        }
    }
}