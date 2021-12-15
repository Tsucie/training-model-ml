using Microsoft.ML.Data;

namespace training_model_ml.Models
{
    // id;date;price;bedrooms;bathrooms;sqft_living;sqft_lot;floors;waterfront;view;condition;grade;sqft_above;sqft_basement;yr_built;yr_renovated;zipcode;lat;long;sqft_living15;sqft_lot15
    public class HouseData
    {
        [LoadColumn(3)]
        public float Label;

        [LoadColumn(4)]
        public float Bedrooms;

        [LoadColumn(5)]
        public float Bathrooms;

        [LoadColumn(6)]
        public float LivingArea;

        [LoadColumn(7)]
        public float LotArea;

        [LoadColumn(8)]
        public float Floors;

        [LoadColumn(9)]
        public float Waterfront;

        [LoadColumn(10)]
        public float View;

        [LoadColumn(11)]
        public float Condition;

        [LoadColumn(12)]
        public float Grade;

        [LoadColumn(13)]
        public float HighFeet;

        [LoadColumn(14)]
        public float BasementFeet;
    }

    public class HousePrediction
    {
        [ColumnName("Score")]
        public float SoldPrice;
    }
}