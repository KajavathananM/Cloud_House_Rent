# main.py
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field

# Request Data Schema
class RentValuationRequest(BaseModel):
    sqft: float = Field(..., gt=0, description="Square footage of the property")
    bedrooms: int = Field(..., ge=1, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, description="Number of bathrooms")
    dist_city_center: float = Field(..., ge=0, description="Distance to city center in km")
    has_balcony: bool = Field(False, description="Whether the unit has a balcony")
    has_parking: bool = Field(False, description="Whether dedicated parking is available")

# Response Data Schema
class RentValuationResponse(BaseModel):
    status: str
    estimated_rent: float
    currency: str = "USD"

# Lifespan manager to load artifacts ONCE on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global interpreter, input_details, output_details, means, stds
    
    # 1. Load feature normalization parameters
    with open("scaler_params.json", "r") as f:
        scaler_data = json.load(f)
        means = np.array(scaler_data["mean"], dtype=np.float32)
        stds = np.array(scaler_data["std"], dtype=np.float32)

    # 2. Load TFLite Model & Allocate Tensors
    interpreter = tf.lite.Interpreter(model_path="rental_valuation_quant.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("FastAPI: TFLite model and scaler parameters loaded into memory.")
    yield
    # Cleanup on shutdown
    interpreter = None


app = FastAPI(
    title="Rental Price Estimator API",
    description="Backend service for mobile housing application",
    version="1.0.0"
    lifespan=lifespan
)


@app.get("/")
def read_root():
    return {"message": "Rental Estimator API is running"}

@app.post("/estimate", response_model=RentValuationResponse)
def predict_rent(data: RentValuationRequest):
    if interpreter is None or means is None or stds is None:
        raise HTTPException(status_code=500, detail="Server model artifacts not loaded.")

    try:
        # Construct feature vector matching training feature order
        raw_features = np.array([
            data.sqft,
            float(data.bedrooms),
            float(data.bathrooms),
            data.dist_city_center,
            1.0 if data.has_balcony else 0.0,
            1.0 if data.has_parking else 0.0
        ], dtype=np.float32)

        # Standard Scaling: (raw_value - mean) / std
        scaled_features = (raw_features - means) / stds
        
        # Reshape to 2D tensor shape [1, 6]
        input_tensor = np.expand_dims(scaled_features, axis=0)

        # Set tensor input and invoke TFLite interpreter
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        # Fetch output value from output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])
        estimated_price = float(prediction[0][0])

        return RentValuationResponse(
            status="success",
            estimated_rent=round(estimated_price, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Make sure the file is named main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)