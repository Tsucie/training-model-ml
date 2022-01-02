using System;
using System.Net;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http;
using training_model_ml.Models;

namespace training_model_ml.Controllers
{
    /// <summary>
    /// Controller untuk menerima api yang akan memprediksi harga dari dataset model
    /// </summary>
    [Route("api/[controller]")]
    [ApiController]
    public class PricePredictionController : ControllerBase
    {
        private string pathData = Startup.rootPath + "/Data/";
        private string trainedPathData = Startup.rootPath + "/TrainedData/";

        /// <summary>
        /// Menjalankan logika linear regression untuk memprediksi harga
        /// </summary>
        [HttpPost("PredictModel")]
        public ActionResult RunPredictionAlgorithm([FromForm] string modelFilename, [FromForm] char separator)
        {
            ResponseMessage resmsg = new ResponseMessage();
            try
            {
                // Validation
                if (string.IsNullOrEmpty(modelFilename) || !char.IsSeparator(separator))
                    throw new Exception("", new Exception("masukan nama file dataset dan separator yang benar"));
                
                // Buat environment ML.NET
                MLContext mLContext = new MLContext(0);
                // Menentukan path Dataset model
                string dataset = this.pathData + modelFilename;
                // Menentukan path untuk menyimpan trained dataset
                string trainedDataset = this.trainedPathData + "Trained" + modelFilename;
                // Menjalankan algoritma machine learning linear regression menggunakan pipeline
                HousePriceModel.CreateModelUsingPipeline(mLContext, dataset, trainedDataset, separator);

                resmsg.Code = 1;
                resmsg.Message = $"Prediksi harga telah ditambahkan ke dataset, cek {trainedDataset}";
                return Ok(resmsg);
            }
            catch (Exception ex)
            {
                resmsg.Error(ex);
                return BadRequest(resmsg);
            }
        }
    }
}