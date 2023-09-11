import React, { useEffect, useState, useRef } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';


const LABELS = {
  "The Eiffel Tower": 0, "The Mona Lisa": 1, "airplane": 2,
  "ambulance": 3, "apple": 4, "axe": 5,
  "banana": 6, "basketball": 7, "bed": 8,
  "butterfly": 9, "cake": 10, "fish": 11,
  "mushroom": 12, "pants": 13, "shoe": 14,
  "star": 15, "t-shirt": 16, "television": 17, "umbrella": 18
};


function App() {
  const canvasRef = useRef(null);
  const [ctx, setCtx] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [prevX, setPrevX] = useState(0);
  const [prevY, setPrevY] = useState(0);
  const [undoList, setUndoList] = useState([]);
  const [redoList, setRedoList] = useState([]);
  const [model, setModel] = useState(null);
  const [finalText, setFinalText] = useState("...");


  function argmax(arr) {
    let maxIndex = 0;
    let maxValue = arr[0];

    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > maxValue) {
        maxIndex = i;
        maxValue = arr[i];
      }
    }

    return maxIndex;
  }

  function getTextForIndex(index) {
    const keys = Object.keys(LABELS);
    const text = keys.find(key => LABELS[key] === index);
    return text;
  }

  useEffect(() => {
    const disableTouchScrolling = (evt) => {
      if (drawing) {
        evt.preventDefault();
      }
    };

    window.addEventListener('touchmove', disableTouchScrolling, { passive: false });

    return () => {
      window.removeEventListener('touchmove', disableTouchScrolling);
    };
  }, [drawing]);






  useEffect(() => {


    async function loadModel() {
      console.log("Loading model...");

      const loadedModel = await tf.loadLayersModel('./model.json');
      console.log("Model loaded:", loadedModel);
      setModel(loadedModel);
    }

    loadModel();

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
    setCtx(context);
  }, []);

  useEffect(() => {
    const intervalId = setInterval(async () => {
      if (model) {
        const canvas = canvasRef.current;
        const inputArray = await preprocessCanvasImage(canvas);
        const prediction = await model.predict(inputArray).data();
        console.log("Prediction:", prediction);
        const index = argmax(prediction);
        setFinalText(getTextForIndex(index));
      }
    }, 100);

    return () => clearInterval(intervalId);
  }, [model]);

  const preprocessCanvasImage = async (canvas) => {
    const canvasImage = new Image();
    canvasImage.src = canvas.toDataURL('image/png');


    await new Promise((resolve) => {
      canvasImage.onload = resolve;
    });

    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;
    const context = resizedCanvas.getContext('2d');
    context.drawImage(canvasImage, 0, 0, 28, 28);


    // const link = document.createElement('a');
    // link.href = resizedCanvas.toDataURL('image/png');
    // link.download = 'drawing.png';
    // link.click();

    const imageData = context.getImageData(0, 0, 28, 28);
    console.log("jh", imageData);
    const inputArray = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      inputArray[i] = (imageData.data[i * 4] + imageData.data[i * 4 + 1] + imageData.data[i * 4 + 2]) / 3;
    }
    console.log("DF", inputArray);
    const inputTensor = tf.tensor(inputArray).reshape([1, 784]);
    return inputTensor;

  };


  const saveState = () => {
    if (ctx) {
      const canvasCopy = canvasRef.current.cloneNode(false);
      canvasCopy.getContext('2d').drawImage(canvasRef.current, 0, 0);
      setUndoList([...undoList, canvasCopy.toDataURL()]);
      setRedoList([]);
    }
  };

  const undo = () => {
    if (undoList.length > 0) {
      const lastState = undoList[undoList.length - 1];
      const canvasCopy = canvasRef.current.cloneNode(false);
      canvasCopy.getContext('2d').drawImage(canvasRef.current, 0, 0);
      setRedoList([...redoList, canvasCopy.toDataURL()]);
      undoList.pop();
      const img = new Image();
      img.src = lastState;
      img.onload = () => {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.drawImage(img, 0, 0);
      };
    }
  };

  const redo = () => {
    if (redoList.length > 0) {
      const nextState = redoList[redoList.length - 1];
      const canvasCopy = canvasRef.current.cloneNode(false);
      canvasCopy.getContext('2d').drawImage(canvasRef.current, 0, 0);
      setUndoList([...undoList, canvasCopy.toDataURL()]);
      redoList.pop();
      const img = new Image();
      img.src = nextState;
      img.onload = () => {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.drawImage(img, 0, 0);
      };
    }
  };

  const clearCanvas = () => {
    if (ctx) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      setUndoList([]);
      setRedoList([]);
      // Add code to redraw the background
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };


  const handleMouseDown = (evt) => {
    if (ctx) {
      const rect = canvasRef.current.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const y = evt.clientY - rect.top;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 14;
      setDrawing(true);
      setPrevX(x);
      setPrevY(y);
      saveState();
    }
  };

  const handleMouseMove = (evt) => {
    if (drawing && ctx) {
      const rect = canvasRef.current.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const y = evt.clientY - rect.top;
      ctx.lineTo(x, y);
      ctx.stroke();
      setPrevX(x);
      setPrevY(y);
    }
  };

  const handleMouseUp = () => {
    if (drawing) {
      setDrawing(false);
    }
  };

  // Add touch event handlers
  const handleTouchStart = (evt) => {
    evt.preventDefault();
    const rect = canvasRef.current.getBoundingClientRect();
    const x = evt.touches[0].clientX - rect.left;
    const y = evt.touches[0].clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 11;
    setDrawing(true);
    setPrevX(x);
    setPrevY(y);
    saveState();
  };

  const handleTouchMove = (evt) => {
    evt.preventDefault();
    if (drawing && ctx) {
      const rect = canvasRef.current.getBoundingClientRect();
      const x = evt.touches[0].clientX - rect.left;
      const y = evt.touches[0].clientY - rect.top;
      ctx.lineTo(x, y);
      ctx.stroke();
      setPrevX(x);
      setPrevY(y);
    }
  };

  const handleTouchEnd = (evt) => {
    evt.preventDefault();
    if (drawing) {
      setDrawing(false);
    }
  };




  return (
    <div className="App">
      <div className="container">
        <div className="text-container">
          <h2>DRAW AI</h2>
          <p>
            You are trying to draw{' '}
            <br></br>
            <span style={{ fontWeight: 'bold' }}>
              {finalText.charAt(0).toUpperCase() + finalText.slice(1)}
            </span>
          </p>
          <div id="controllers">
            <button className="controller undo" onClick={undo}>
              Undo
            </button>
            <button className="controller redo" onClick={redo}>
              Redo
            </button>
            <button className="controller clear" onClick={clearCanvas}>
              Clear
            </button>

          </div>
        </div>
        <div className="canvas-container">
          <canvas
            width="440"
            height="440"
            id="paint"
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
          ></canvas>

        </div>
      </div>
    </div>
  );

}

export default App;



