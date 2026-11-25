import express from "express"
import dotenv from "dotenv"
import { GoogleGenAI } from "@google/genai"

import { ChatGoogle } from "@langchain/google";
import { ChatGroq } from "@langchain/groq"
import { Annotation, MemorySaver, MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";


dotenv.config()
const app = express()
const port = 5000
app.use(express.json())

//--------- without langchain -------


// const ai = new GoogleGenAI({
//     apiKey: process.env.GEMINI_API_KEY
// })


// app.post("/ai", async (req, res) => {
//     const { input } = req.body
//     try {
//         const interaction = await ai.interactions.create({
//             model: "gemini-3.5-flash",
//             system_instruction: "You are Jarvis, a professional AI assistant.",
//             input:input,
//         });

//         return res.status(200).json({
//             ai : interaction.output_text
//         })
//     } catch (error) {
//         console.log("Gemini response error:",error)
//         return res.status(500).json({
//             message:"AI request failed",
//             error:error.message
//         })

//     }

// })



//--------- with langchain -------

// const model = new ChatGroq({
//     model:"openai/gpt-oss-120b",
//     maxOutputTokens: 300,
//     temperature:1.5,
// })

// app.post("/ai", async (req, res) => {

//     const { input } = req.body
//     try {

//     const response = await model.invoke([
//         ["system","You are Jarvis, a professional AI assistant."],
//         ["human",input]
//     ]);

//     return res.status(200).json({
//         ai: response.text
//     })

//     } catch (error) {
//         console.error("Groq Error:", error);

//         return res.status(500).json({
//             message: "AI request failed"
//         });
//     }

// })



const State = Annotation.Root({
    prompt: Annotation,
    aiMsg: Annotation
})

const tool = new TavilySearch({
    maxResults: 5,
    topic: "general",
});

// creates cache memory
const checkPointer = new MemorySaver()

const tools = [tool]

const toolNode = new ToolNode(tools)

const model = new ChatGroq({
    model: "openai/gpt-oss-120b",
    maxOutputTokens: 300,
    temperature: 1.5,
}).bindTools(tools)


const callLLM = async (state) => {
    const response = await model.invoke([
        ["system", "You are Jarvis, a professional AI assistant.If you don,t have content regarding prompt then don't give respone"],
        ...state.messages
    ]);

    return { messages: [response] }
}

const shouldContinue = async (state) => {
    const aiMessage = state.messages[state.messages.length - 1]
    if (aiMessage.tool_calls.length > 0) return "tools"

    return "__end__"
}

// const graph=new StateGraph(State)
// .addNode("agent",callLLM)
// .addNode("tools",toolNode)
// .addEdge("__start__","agent")
// .addEdge("tools","agent")
// .addConditionalEdges("agent",shouldContinue)
// .compile()

const graph = new StateGraph(MessagesAnnotation)
    .addNode("agent", callLLM)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addEdge("tools", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .compile({ checkpointer: checkPointer })


app.post("/ai", async (req, res) => {

    const { input } = req.body
    try {

        const response = await graph.invoke({
            messages: [
                ["human", input]
            ]
        },{
            configurable:{thread_id:"user123"}
        });
        console.log(response.messages)

        return res.status(200).json({
            ai: response.messages[response.messages.length - 1].content
        })

    } catch (error) {
        console.error("Groq Error:", error);

        return res.status(500).json({
            message: "AI request failed"
        });
    }

})


app.get("/", (req, res) => {
    return res.json({
        message: "hello from level-1"
    })
})

app.listen(port, () => {
    console.log("server started")
})
