
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import pysd
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import networkx as nx

app = FastAPI()

model = None
results = None

@app.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    global model, results
    model = None
    results = None
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        if ext == ".mdl":
            model = pysd.read_vensim(tmp_path)
        elif ext == ".xmile":
            model = pysd.read_xmile(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Model {file.filename} loaded successfully."}

@app.get("/model_report")
def generate_model_report():
    if not model:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        # Extract model structure
        stocks = [comp for comp in dir(model.components) if isinstance(getattr(model.components, comp), pysd.core.Stock)]
        flows = [comp for comp in dir(model.components) if isinstance(getattr(model.components, comp), pysd.core.Flow)]
        auxiliaries = [comp for comp in dir(model.components) if isinstance(getattr(model.components, comp), pysd.core.Auxiliary)]

        report = {
            "stocks": stocks,
            "flows": flows,
            "auxiliaries": auxiliaries
        }

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/simulation_summary")
def simulation_summary():
    global results
    if results is None:
        raise HTTPException(status_code=400, detail="No simulation results available")

    try:
        # Provide key simulation results summary
        summary = {
            "final_time": results.index[-1],
            "stocks": {col: results[col].iloc[-1] for col in results.columns if col in model.stocks},
            "flows": {col: results[col].iloc[-1] for col in results.columns if col in model.flows}
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating simulation summary: {str(e)}")

@app.get("/feedback_loops")
def identify_feedback_loops():
    if not model:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        # Generate the feedback loop structure
        g = nx.DiGraph()
        for eqn in dir(model.components):
            if eqn.startswith("_"):
                continue
            try:
                expr = getattr(model.components, eqn).equation
                if isinstance(expr, list):
                    for var in expr:
                        if isinstance(var, str):
                            g.add_edge(var, eqn)
            except Exception:
                continue

        loops = list(nx.simple_cycles(g))
        if not loops:
            return {"message": "No causal loops detected in model."}

        loop_analysis = [{"loop": loop, "type": "positive" if len(loop) % 2 == 0 else "negative"} for loop in loops]

        return {"feedback_loops": loop_analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error identifying feedback loops: {str(e)}")

@app.get("/sd_report")
def generate_sd_report():
    model_report = generate_model_report()
    simulation_summary = simulation_summary()
    feedback_loops = identify_feedback_loops()

    # Combine all reports into a single SD report
    report = {
        "model_structure": model_report,
        "simulation_summary": simulation_summary,
        "feedback_loops": feedback_loops
    }

    return report

@app.get("/visualize")
def visualize_results():
    global results
    if results is None:
        raise HTTPException(status_code=400, detail="No simulation results available")

    try:
        fig, ax = plt.subplots()
        results.plot(ax=ax)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return {"image_base64": encoded, "message": "Visualization generated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/convert")
def convert_to_python():
    if not model:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        py_model_path = model.py_model_file
        return FileResponse(py_model_path, media_type='text/plain', filename=os.path.basename(py_model_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sfd")
def generate_stock_flow_diagram():
    if not model:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        g = nx.DiGraph()
        for eqn in dir(model.components):
            if eqn.startswith("_"):
                continue
            try:
                expr = getattr(model.components, eqn).equation
                if isinstance(expr, list):
                    for var in expr:
                        if isinstance(var, str):
                            g.add_edge(var, eqn)
            except Exception:
                continue
        if g.number_of_nodes() == 0:
            return {"message": "No diagram structure could be inferred."}

        pos = nx.spring_layout(g)
        fig, ax = plt.subplots()
        nx.draw(g, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8, ax=ax)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return {"image_base64": encoded, "message": "Stock & Flow diagram generated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagram error: {str(e)}")

@app.get("/cld")
def generate_causal_loop_diagram():
    if not model:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        g = nx.DiGraph()
        for eqn in dir(model.components):
            if eqn.startswith("_"):
                continue
            try:
                expr = getattr(model.components, eqn).equation
                if isinstance(expr, list):
                    for var in expr:
                        if isinstance(var, str):
                            g.add_edge(var, eqn)
            except Exception:
                continue

        loops = list(nx.simple_cycles(g))
        if not loops:
            return {"message": "No causal loops detected in model."}
        return {"loops": loops}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loop detection error: {str(e)}")

@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return get_openapi(
        title="PySD API",
        version="1.0",
        description="API for uploading, simulating, and visualizing system dynamics models.",
        routes=app.routes
    )
