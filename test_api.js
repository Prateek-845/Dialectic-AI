const axios = require('axios');

async function test() {
  try {
    const response = await axios({
      method: 'post',
      url: 'https://dialectic-ai-engine.onrender.com/analyze/stream',
      data: { article: "test", thread_id: "test" },
      responseType: 'stream'
    });
    console.log("Status:", response.status);
    response.data.on('data', chunk => console.log(chunk.toString()));
  } catch (error) {
    console.log("Error:", error.message);
    if (error.response) {
      console.log("Status:", error.response.status);
      console.log("Data:", error.response.data);
    }
  }
}

test();
