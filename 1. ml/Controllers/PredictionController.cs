using Microsoft.AspNetCore.Mvc;

namespace HomePricesWebApp.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PredictionController : ControllerBase
{
    private readonly HomePricePredictor _predictor;

    public PredictionController(HomePricePredictor predictor)
    {
        _predictor = predictor;
    }

    [HttpPost("predict")]
    public ActionResult<PredictionResponse> Predict([FromBody] PredictionRequest request)
    {
        try
        {
            var predictedPrice = _predictor.PredictPrice(
                request.OverallQual,
                request.GrLivArea,
                request.GarageCars,
                request.TotalSquareFeet,
                request.TotalBath,
                request.SqFtPerRoom,
                request.YearBuilt
            );

            return Ok(new PredictionResponse
            {
                PredictedPrice = predictedPrice,
                Success = true
            });
        }
        catch (Exception ex)
        {
            return BadRequest(new PredictionResponse
            {
                PredictedPrice = "",
                Success = false,
                Error = ex.Message
            });
        }
    }

    [HttpGet("health")]
    public ActionResult<object> Health()
    {
        return Ok(new { status = "healthy", timestamp = DateTime.UtcNow });
    }
}