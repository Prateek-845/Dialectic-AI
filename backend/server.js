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
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

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


app.post('/api/auth/register', async (req, res) => {
  try {
    const { username, password } = req.body;
    if (!username || !password) return res.status(400).json({ error: 'Username and password required' });
    
    const existingUser = await User.findOne({ username });
    if (existingUser) return res.status(400).json({ error: 'Username already taken' });

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ username, password: hashedPassword });
    await newUser.save();

    const token = jwt.sign({ id: newUser._id, username: newUser.username }, JWT_SECRET, { expiresIn: '24h' });
    res.status(201).json({ token, username: newUser.username });
  } catch (error) {
    res.status(500).json({ error: 'Registration failed' });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    const user = await User.findOne({ username });
    if (!user) return res.status(400).json({ error: 'Invalid username or password' });

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) return res.status(400).json({ error: 'Invalid username or password' });

    const token = jwt.sign({ id: user._id, username: user.username }, JWT_SECRET, { expiresIn: '24h' });
    res.json({ token, username: user.username });
  } catch (error) {
    res.status(500).json({ error: 'Login failed' });
  }
});


app.post('/api/debates/stream', authenticateToken, async (req, res) => {
  try {
    const { article, thread_id, action, jury_feedback } = req.body;
    if (!article && !thread_id) return res.status(400).json({ error: 'Article or thread_id is required' });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const response = await axios({
      method: 'post',
      url: `${PYTHON_API_URL}/analyze/stream`,
      data: { article: article || "", thread_id, action, jury_feedback },
      responseType: 'stream'
    });

    let finalState = null;


    response.data.pipe(res);

    response.data.on('data', (chunk) => {
      const chunkStr = chunk.toString();
      const lines = chunkStr.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ') && !line.includes('[DONE]')) {
          try {
            finalState = JSON.parse(line.substring(6));
          } catch(e) {}
        }
      }
    });

    response.data.on('error', (err) => {
      console.error('Stream error from Python Engine:', err.message);
      if (!res.headersSent) {
        res.status(500).end();
      } else {
        res.end();
      }
    });

    response.data.on('end', async () => {
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
        } catch(e) {
          console.error("Error saving streamed debate to MongoDB", e);
        }
      }
    });
  } catch (error) {
    console.error('Error streaming debate:', error.message);
    res.status(500).json({ error: 'Failed to stream debate' });
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
});
