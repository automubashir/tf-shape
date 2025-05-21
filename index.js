const express = require('express');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');

// Use pure JavaScript version of TensorFlow.js
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu');

const port = 4444;
const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(bodyParser.json({ limit: '10mb' }));
app.use(express.static('public'));
app.use(express.static('data'));

// File paths for model and metadata
const MODEL_DIR = path.join(__dirname, 'data');
const MODEL_WEIGHTS_PATH = path.join(MODEL_DIR, 'model-weights.bin');
const MODEL_JSON_PATH = path.join(MODEL_DIR, 'model.json');
const PROMPTS_PATH = path.join(MODEL_DIR, 'prompts.json');

let model;
let prompts = [];

// Custom handler for saving and loading models with pure JS TensorFlow
class FileSyncHandler {
  constructor() {}

  // Save model to disk
  async save(modelArtifacts) {
    if (!fs.existsSync(MODEL_DIR)) {
      fs.mkdirSync(MODEL_DIR, { recursive: true });
    }

    // Save model topology and weights separately
    fs.writeFileSync(MODEL_JSON_PATH, JSON.stringify(modelArtifacts.modelTopology));

    // Save weights as binary file
    const weightData = modelArtifacts.weightData;
    const buffer = Buffer.from(weightData);
    fs.writeFileSync(MODEL_WEIGHTS_PATH, buffer);

    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON'
      }
    };
  }

  // Load model from disk
  async load() {
    if (!fs.existsSync(MODEL_JSON_PATH) || !fs.existsSync(MODEL_WEIGHTS_PATH)) {
      throw new Error('Model files not found');
    }

    // Load model topology
    const modelTopology = JSON.parse(fs.readFileSync(MODEL_JSON_PATH, 'utf8'));

    // Load weight data
    const weightData = fs.readFileSync(MODEL_WEIGHTS_PATH);

    // Convert Node.js buffer to ArrayBuffer
    const weightDataArrayBuffer = new Uint8Array(weightData).buffer;

    // Return model artifacts
    return {
      modelTopology,
      weightData: weightDataArrayBuffer
    };
  }
}

// One-hot encode prompts
function encodePrompt(prompt, labels) {
  const index = labels.indexOf(prompt);
  const oneHot = Array(labels.length).fill(0);
  if (index >= 0) oneHot[index] = 1;
  return oneHot;
}

async function loadOrTrainModel() {
  try {
    // Try to load prompts first
    if (fs.existsSync(PROMPTS_PATH)) {
      prompts = JSON.parse(fs.readFileSync(PROMPTS_PATH, 'utf8'));
      console.log('✅ Prompts loaded from file.', prompts);
    } else {
      throw new Error('Prompts file not found');
    }

    // Try to load model using custom handler
    if (fs.existsSync(MODEL_JSON_PATH) && fs.existsSync(MODEL_WEIGHTS_PATH)) {
      const handler = new FileSyncHandler();
      const artifacts = await handler.load();

      // Reconstruct model from topology
      model = await tf.loadLayersModel(tf.io.fromMemory(artifacts));
      console.log('✅ Model loaded from file.');
    } else {
      throw new Error('Model files not found');
    }
  } catch (e) {
    console.log('⚠️ Model not found or incomplete. Training new model...', e.message);

    // Load dataset
    const dataPath = path.join(__dirname, 'data', 'shapes.json');
    if (!fs.existsSync(dataPath)) {
      console.error('❌ Training data not found at:', dataPath);
      process.exit(1);
    }

    const raw = fs.readFileSync(dataPath);
    const dataset = JSON.parse(raw);

    prompts = [...new Set(dataset.map((d) => d.prompt))];
    console.log('Found prompts:', prompts);

    const inputs = dataset.map((d) => encodePrompt(d.prompt, prompts));

    const inputTensor = tf.tensor2d(inputs);
    const MAX_POINTS = 27;

    function normalizePoints(points) {
      const flat = points.flat();
      const needed = MAX_POINTS * 2;

      if (flat.length > needed) {
        return flat.slice(0, needed); // Truncate
      } else {
        return flat.concat(Array(needed - flat.length).fill(0)); // Pad with 0s
      }
    }

    const outputs = dataset.map((d) => normalizePoints(d.points));
    const outputTensor = tf.tensor2d(outputs);

    // Create model
    model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [prompts.length],
        units: 64,
        activation: 'relu',
      })
    );
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: outputs[0].length })); // Output shape matches flat points

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    console.log('Training model with', dataset.length, 'examples...');
    await model.fit(inputTensor, outputTensor, {
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 10 === 0) {
            console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}`);
          }
        }
      }
    });

    // Make sure directory exists
    if (!fs.existsSync(MODEL_DIR)) {
      fs.mkdirSync(MODEL_DIR, { recursive: true });
    }

    // Save prompts
    fs.writeFileSync(PROMPTS_PATH, JSON.stringify(prompts));
    console.log('✅ Prompts saved to file.');

    // Save model using custom handler
    const handler = new FileSyncHandler();
    await model.save(tf.io.withSaveHandler(handler.save.bind(handler)));
    console.log('✅ New model trained and saved.');

    // Clean up tensors
    inputTensor.dispose();
    outputTensor.dispose();
  }
}

// Add a route to check available prompts
app.get('/prompts', (req, res) => {
  res.json({ prompts });
});

// Add a route to check if server is running
app.get('/status', (req, res) => {
  res.json({
    status: 'ok',
    modelLoaded: !!model,
    promptsLoaded: prompts.length > 0,
    prompts
  });
});

// Add endpoints for dataset management
app.post('/save-dataset', (req, res) => {
  try {
    const dataset = req.body;

    // Validate dataset
    if (!Array.isArray(dataset)) {
      return res.status(400).json({ error: 'Invalid dataset format - must be an array' });
    }

    // Create directory if it doesn't exist
    if (!fs.existsSync(MODEL_DIR)) {
      fs.mkdirSync(MODEL_DIR, { recursive: true });
    }

    // Save dataset to file
    fs.writeFileSync(path.join(MODEL_DIR, 'shapes.json'), JSON.stringify(dataset, null, 2));

    res.json({ success: true, message: 'Dataset saved successfully' });
  } catch (error) {
    console.error('Error saving dataset:', error);
    res.status(500).json({ error: 'Failed to save dataset: ' + error.message });
  }
});

app.post('/predict', async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'Missing prompt parameter' });
    }

    // Use the separately loaded prompts
    const encoded = tf.tensor2d([encodePrompt(prompt, prompts)]);
    const prediction = model.predict(encoded);
    const output = await prediction.array();

    // Clean up tensors to prevent memory leaks
    encoded.dispose();
    prediction.dispose();

    // Convert flat array back to points
    const points = [];
    for (let i = 0; i < output[0].length; i += 2) {
      if (output[0][i] !== 0 || output[0][i + 1] !== 0) { // Skip padding zeros
        points.push([output[0][i], output[0][i + 1]]);
      }
    }

    res.json({ points });
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

app.post('/retrain', async (req, res) => {
  try {
    // Force retraining by loading model with training flag
    await loadOrTrainModel(true);
    res.json({ success: true, message: 'Model retrained successfully' });
  } catch (error) {
    console.error('Error retraining model:', error);
    res.status(500).json({ error: 'Failed to retrain model: ' + error.message });
  }
});

// Start server
app.listen(port, async () => {
  console.log('🚀 Server starting on http://localhost:' + port);
  try {
    await loadOrTrainModel();
    console.log('✅ Server ready for predictions');
  } catch (err) {
    console.error('❌ Failed to initialize model:', err.message);
    process.exit(1);
  }
});