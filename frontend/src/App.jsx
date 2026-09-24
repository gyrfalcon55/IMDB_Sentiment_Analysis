import React, { useState } from 'react'

function App() {

  const [input, setInput] = useState("")
  const [sentiment, setsentiment] = useState(
    {
      text: ""
    })

  const sendMessage = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: JSON.stringify({
          text: input,
        })
      });

      const data = await response.json()

      console.log(data);

      setsentiment({
        text: data.result,
      })

      setInput("");
    } catch (error) {
      console.error("Error:", error)
    }
  }





  return (
    <div className="bg-[url('/my_img.jpg')] bg-cover bg-center bg-no-repeat w-full min-h-screen flex justify-center items-center">

      <div
        id="container"
        className="w-lg h-100 rounded-3xl
             bg-white/5
             backdrop-blur-md
             border border-white/50
             shadow-xl
             justify-items-center
             content-center
             text-amber-50
             "

      >

        <div className="bg-[url('/imdb_logo.png')] bg-cover bg-center bg-no-repeat w-40 h-20"></div>
        <h1 className='flex m-5 text-2xl'>IMDB Movie Review Sentiment Prediction</h1>
        <input
          id='inputname'
          type="text"
          placeholder='type....'
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="w-sm h-15 border-2 border-sky-500 m-8 flex rounded-xl"
        />

        <button onClick={sendMessage} className='text-lg font-bold text-black m-8 w-sm justify-self-center flex bg-yellow-500 hover:bg-yellow-200 rounded-xl justify-center'>Submit</button>
        <div className="text-xl text-white">
          Sentiment -{" "}
          <span
            className={
              sentiment.text === "Positive"
                ? "text-green-500"
                : sentiment.text === "Negative"
                  ? "text-red-500"
                  : "text-white"
            }
          >
            {sentiment.text || "Not predicted"}
          </span>
        </div>
      </div>




    </div >
  )
}

export default App