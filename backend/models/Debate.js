const mongoose = require('mongoose');

const DebateSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  original_article: String,
  persona_a: String,
  persona_b: String,
  final_summary: String,
  debate_log: [mongoose.Schema.Types.Mixed], // Store the flexible round data
  synthesis_rouge: mongoose.Schema.Types.Mixed,
  synthesis_neutral: mongoose.Schema.Types.Mixed,
  debate_influence: mongoose.Schema.Types.Mixed,
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Debate', DebateSchema);