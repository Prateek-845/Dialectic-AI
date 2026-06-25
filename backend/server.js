require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');
const Debate = require('./models/Debate');
const User = require('./models/User');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'dialectic-super-secret-key';
const app = express();
const PORT = process.env.PORT || 5000;
const rawPythonUrl = process.env.PYTHON_API_URL || 'http://localhost:8000';
const PYTHON_API_URL = rawPythonUrl.replace(/\/+$/, '');

app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/dialectic', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => {
  console.log('Connected to MongoDB');
}).catch((err) => {
  console.error('MongoDB connection error:', err);
});


const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Access denied. Please log in.' });

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: 'Invalid or expired session. Please log in again.' });
    req.user = user;
    next();
  });
};


const signToken = (user, res, status = 200) => {
  const token = jwt.sign({ id: user._id, username: user.username }, JWT_SECRET, { expiresIn: '24h' });
  res.status(status).json({ token, username: user.username });
};

app.post('/api/auth/register', async (req, res) => {
  try {
    const { username, password } = req.body;
    if (!username || !password) return res.status(400).json({ error: 'Credentials required' });
    if (await User.findOne({ username })) return res.status(400).json({ error: 'Username taken' });

    const newUser = await new User({ username, password: await bcrypt.hash(password, 10) }).save();
    signToken(newUser, res, 201);
  } catch (e) { res.status(500).json({ error: 'Registration failed' }); }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    const user = await User.findOne({ username });
    if (!user || !(await bcrypt.compare(password, user.password))) return res.status(400).json({ error: 'Invalid credentials' });
    signToken(user, res);
  } catch (e) { res.status(500).json({ error: 'Login failed' }); }
});


app.post('/api/debates/stream', authenticateToken, async (req, res) => {
  const abortController = new AbortController();
  let responseStream = null;

  res.on('close', () => {
    if (!res.writableEnded) {
      console.log('Client disconnected from Express, aborting Python engine stream request.');
      abortController.abort();
      if (responseStream) {
        try {
          responseStream.destroy();
        } catch (err) {
          console.error('Error destroying response stream:', err.message);
        }
      }
    }
  });

  try {
    const { article, thread_id, action, jury_feedback } = req.body;
    if (!article && !thread_id) return res.status(400).json({ error: 'Article or thread_id is required' });

    console.log(`[Express] Initiating stream request. Target URL: ${PYTHON_API_URL}/analyze/stream`);
    const response = await axios({
      method: 'post',
      url: `${PYTHON_API_URL}/analyze/stream`,
      data: { article: article || "", thread_id, action, jury_feedback },
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/event-stream'
      },
      responseType: 'stream',
      timeout: 60000,
      signal: abortController.signal
    });

    responseStream = response.data;

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    let finalState = null;


    responseStream.pipe(res);

    let buffer = '';
    responseStream.on('data', (chunk) => {
      buffer += chunk.toString();
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        if (part.startsWith('data: ') && !part.includes('[DONE]')) {
          try {
            finalState = JSON.parse(part.substring(6));
          } catch (e) { }
        }
      }
    });

    responseStream.on('error', (err) => {
      console.error('Stream error from Python Engine:', err.message);
      if (!res.headersSent) {
        res.status(500).end();
      } else {
        res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
        res.end();
      }
    });

    responseStream.on('end', async () => {
      res.end();

      if (finalState && finalState.final_summary) {
        try {
          const newDebate = new Debate({
            userId: req.user.id,
            original_article: finalState.original_article,
            persona_a: finalState.persona_a,
            persona_b: finalState.persona_b,
            final_summary: finalState.final_summary,
            debate_log: finalState.debate_log || [],
            synthesis_rouge: finalState.synthesis_rouge,
            synthesis_neutral: finalState.synthesis_neutral,
            debate_influence: finalState.debate_influence
          });
          await newDebate.save();
        } catch (e) {
          console.error("Error saving streamed debate to MongoDB", e);
        }
      }
    });
  } catch (error) {
    if (axios.isCancel(error)) {
      console.log('Axios request was canceled due to client disconnect.');
      return;
    }
    console.error('Error streaming debate:', error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: `AI Engine may be waking up from sleep. Please try again in 60s. (Error: ${error.message})` });
    } else {
      res.write(`data: ${JSON.stringify({ error: `AI Engine Error: ${error.message}` })}\n\n`);
      res.end();
    }
  }
});



app.get('/api/debates', authenticateToken, async (req, res) => {
  try {
    const debates = await Debate.find({ userId: req.user.id }).sort({ createdAt: -1 });
    res.json(debates);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch debates' });
  }
});

app.get('/api/debates/:id', async (req, res) => {
  try {
    const debate = await Debate.findById(req.params.id);
    if (!debate) {
      return res.status(404).json({ error: 'Debate not found' });
    }
    res.json(debate);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch debate' });
  }
});

app.listen(PORT, () => {
  console.log(`Express Server running on port ${PORT}`);
  console.log(`Using PYTHON_API_URL: "${PYTHON_API_URL}"`);
});
