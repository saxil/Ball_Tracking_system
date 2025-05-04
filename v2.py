from roboflow import Roboflow

rf = Roboflow(api_key="NEcWAeRre56Q75VTFXWg")
project = rf.workspace().project("green-ball-v17ti-arnob")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "test.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)