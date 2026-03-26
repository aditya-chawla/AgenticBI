**Agentic BI- 295A Final Report**

| A Project ReportPresented to The Faculty of the College ofEngineering |
| :---: |
| San Jose State UniversityIn Partial FulfillmentOf the Requirements for the Degree**Master of Science in Artificial Intelligence** |
| By |
| Apoorva Adimulam, Aditya Chawla, Kushagra Kshatri, Shivang Patel |
| May 2026 |

| Copyright © 2026 |
| :---: |
| Apoorva Adimulam, Aditya Chawla, Kushagra Kshatri, Shivang Patel |
| ALL RIGHTS RESERVED |

| APPROVED   |
| ----- |
|   |
|  Dr. Bernardo Flores, Project Advisor |

ABSTRACT  
Agentic BI  
By Apoorva Adimulam, Aditya Chawla, Kushagra Kshatri, Shivang Patel

Data Analysis serves as the foundation of Business Intelligence by providing systematic methods that enable evidence-based decision making for formulating corporate strategies and data-driven actions to solve problems at hand. It allows companies to identify latest trends, monitor performance indicators, and satisfy the ever volatile customer needs, giving companies a significant competitive advantage in fast-paced markets.  
While data analysis provides significant upsides, it is a difficult skill to master and requires analysts to possess significant domain-specific expertise. While analysts are equipped with data visualization and dashboarding tools like PowerBI, Tableau and Redash, they also present some notable challenges. These platforms can be cumbersome to use and are heavily reliant on extensive hands-on experience and manual setup.  Query generation on these platforms requires in-depth understanding of databases, development of visually appealing dashboards is time consuming, getting meaningful insights from these visualizations requires experience, KPI (Key Performance Indicator) monitoring and performance tracking is tedious.   
In this project, we propose an end-to-end automation of the Business Intelligence process to solve the previously mentioned challenges through a Multi-Agent architecture where each Agent is in charge of a specialized sub-process. The expected result would be an interactive multi-agent AI analytics software that decision-makers can interact with to achieve automated high-quality dashboard visualization, help ease the monitoring process of these dashboards, and generate intelligent and explainable insights. This software aims to simplify data analysis and business insights by reducing complexity, saving time, and making informed decision-making smarter.

| Acknowledgments  |
| :---: |
| The authors are deeply indebted to Professor Bernardo Flores for his invaluable comments and assistance in the preparation of this study. |

**Table of Contents**

[**Chapter 1\.  Project Overview	1**](#project-overview)

[Introduction	1](#1.1.-introduction)

[Proposed Areas of Study and Academic Contribution	1](#1.2.-proposed-areas-of-study-and-academic-contribution)

[Current State of the Art	2](#current-state-of-the-art)

[**Chapter 2\.  Project Architecture	4**](#project-architecture)

[Introduction	4](#2.1.-introduction)

[Architecture Subsystems	6](#2.2.-architecture-subsystems)

**List of Figures**

[Figure 1 \- System Architecture](#bookmark=id.udrrds8yjpr8)	5

Figure 2- Mock User Interface							        7

Figure 3: Sequence Diagram of Data Ingestion and SQL Generation 		        8

Figure 4: Sequence diagram of SQL Execution 					        9

Figure 5: Sequence diagram of Data Visualization 					      10

**List of Tables**

1. #  **Project Overview** {#project-overview}

   1. ## **1.1. Introduction** {#1.1.-introduction}

Business Intelligence (BI) has become indispensable for organizations seeking to compete in data-driven markets. And yet, even with the established availability of powerful tools like PowerBI, Tableau, Redash, it’s clear that many businesses are still finding it difficult to fully leverage the real potential of their own data. A major issue is that these existing platforms tend to demand a significant depth of technical expertise, still rely heavily on the manual crafting of complex queries, and require constant, sometimes tedious, oversight to maintain effectiveness. For companies that don't have the luxury of maintaining a large, specialized analytics team, this dependence directly translates into a few critical problems: high operational costs, increased response times, and missed opportunities in the fast-moving commercial sector. BI techniques are becoming modernised with the rise of LLMs and the subsequent development of LLM-based BI models like SiriusBI but there has not been any industry-level adaptation of LLMs in BI frameworks. With other domains rapidly adapting to the advancements in AI, this seems to be the perfect time to incorporate AI-based automation in Business Intelligence.

This project introduces an Agentic Business Intelligence system that leverages multi-agent architectures to automate the complete BI pipeline—from natural language query interpretation to SQL execution, visualization generation, monitoring, Anomaly detection and insight extraction. By deploying specialized autonomous agents coordinated through frameworks like CrewAI, the system reduces dependency on technical personnel, enables real-time anomaly detection, and scales efficiently without proportional increases in human resources. The architecture integrates guardrails for privacy and security, semantic layers for business logic consistency, and AI-native visualization frameworks \- Vizro to deliver enterprise-grade analytics capabilities.

2. ## **1.2. Proposed Areas of Study and Academic Contribution** {#1.2.-proposed-areas-of-study-and-academic-contribution}

Current BI automation research predominantly explores monolithic LLM applications, yet research evidence suggests complex analytical workflows benefit from self-managed, dynamic resilient systems. This project operationalizes the theoretical principles of muli-agent system through a production-oriented architecture where autonomous agents handle distinct responsibilities—schema understanding, query formulation, execution validation, visualization, monitoring and insight extraction. The research quantifies performance gains of this distributed approach against single-agent baselines, measuring accuracy, semantic correctness, and scalability across the complete analytics pipeline. By implementing Agent-to-Agent (A2A) communication protocols and Model Context Protocol (MCP) integration through frameworks like CrewAI, the work provides replicable design patterns for enterprise agentic BI systems.

This research also addresses critical gaps where current single-agent systems fall short in enterprise data analytics by advancing individual components across the complete BI pipeline—including NL2SQL translation, Text-to-Visualization generation, guardrails for AI agents, multi-agent orchestration, seamless tool calling, and inter-agent communication. The implementation leverages modern approaches such as Schema-Graph with Integrated Syntax (SGIS) for context-aware query generation and the LLM-Enhanced Visual Analytics (LEVA) framework for interactive data exploration, while integrating production-ready enterprise-grade tools like Vizro for AI-native dashboard development. Furthermore, the architecture adopts emerging open standards including Agent-to-Agent (A2A) protocol for direct agent communication and Model Context Protocol (MCP) for structured tool access and resource management, establishing a foundation for state-of-the-art agentic analytics systems.

3. ## **Current State of the Art** {#current-state-of-the-art}

The state-of-the-art tools and technologies in BI and Analytics have gone through a fundamental shift from reactive to proactive intelligence, with leading platforms converging on conversational interfaces, real-time processing capabilities, and autonomous AI systems. This evolution positions business intelligence as the operational nervous system of modern enterprises, democratizing data access through integrated AI-native platforms that eliminate traditional barriers between users, applications and insights.

Microsoft the market leader and technology powerhouse, had released Azure Fabric’s Real-Time Intelligence \[1\] an end-to-end solution for ingesting, processing, analyzing, visualizing, monitoring, alerting, and acting on data which has become the fastest-growing workload with 24,000+ customers and 6x adoption growth in the past year \[2\]. The platform enables users to build real-time dashboards using natural language with Copilot, eliminating technical barriers to real-time analytics. Power BI by Microsoft which is the market leader in BI and analytics industry is geared to be transformed into an AI-powered business intelligence system with Copilot Default-On Experience \[3\] to provide a full-screen, chat-based AI interface that allows users to explore data and generate insights across reports without traditional interface barriers. 

Tableau by salesforce the second most popular BI software in its latest  release focuses on agentic analytics with Enhanced Q\&A with Multilingual Support \[4\] allowing users to ask questions in their preferred language with improved text formatting and multiple entry points. Correlated Metrics Insights provides new AI-driven insight types that automatically identify significant relationships between metrics. 

Redash is a commonly used Open-source BI platform that has been democratizing data access for over a decade, with a SQL oriented approach rather than relying on GUI. Redash offers enterprise-grade capabilities for visualization and dashboarding without licensing constraints thereby offering similar performance for budget constrained companies and is a solid alternative to closed source products available in the market.

Vizro is a modern open source AI native BI framework developed by McKinsey for low-code/no-code approach to create production ready enterprise grade dashboards. It is developed using modern web standards (react and python) and also supports native and deep integration with AI/ML frameworks such as Crew AI, MCP (Model Context Protocol), A2A(Agent to agent protocol) etc. It is a forward-looking framework as a modern alternative to traditional BI tools.       

The business intelligence industry is going through unprecedented transformation, driven by generative AI integration, real-time intelligence capabilities, and conversational analytics. Leading vendors are rapidly deploying cutting-edge features that fundamentally reshape how organizations interact with data.

4. 

2. # **Project Architecture** {#project-architecture}

   1. ## **2.1. Introduction** {#2.1.-introduction}

This Agentic BI system is based on a layered, event-driven, and service-oriented architecture featuring isolation of user interactions, robust agent orchestration, and scalable analytics pipelines. This architecture moves beyond traditional static dashboards by combining conversational AI with enterprise-grade agentic orchestration, thus enabling automation of sophisticated business intelligence workflows all the way from natural language understanding to data retrieval and visualization.

This system is organized into four major logical layers: the UI Layer, the Guardrail Layer, the central Orchestrator, and the Multi-Agent Layer.Given the complexity that needs to be dealt with while building and deploying this project, following such modular architecture allows for high scalability while providing consistency across all operations. Separating the user interface from the agentic processing logic would ensure that complex queries can be handled asynchronously, without interfering with client-side rendering. A guardrail layer on top of this entire architecture enhances the privacy of the overall product.


![][image1]

Fig. 1 \- Project Architecture

Figure 1 gives a high-level visual description of the architecture of Agentic BI. The system adopts a layered, event-driven, and service-oriented architecture designed to isolate user interactions, ensure robust agent orchestration, support scalable analytics pipelines, and maintain strict enterprise-grade security and privacy controls. This architecture embodies the modern paradigms of business intelligence by seamlessly integrating conversational AI and enterprise agentic orchestration, aligning with both current academic research and leading industry deployments. 

Guardrails sits between the UI and core system, providing real-time input validation, output filtering, and enforcement of privacy policies. Implements authentication, authorization, zero-trust practices, privacy controls, and compliance auditing. 

The Agent Orchestration Layer is the heart of the solution and uses the CrewAI orchestration framework for workflow management. It supports Agent-to-Agent (A2A) protocol for direct agent-to-agent communication and manages the lifecycle of individual task-specialized agents. Specialized Agents for individual tasks : the Monitoring Agent handles event streaming, real-time anomaly detection, and efficient alerting via stream processing pipelines. Analytics Agent performs statistical analysis and machine-learning powered insight extraction. Visualization Agent manages chart generation and dashboard components, integrating with modern frameworks like Vizro for low-code, real-time BI. Data Agent is responsible for data discovery, query processing, connecting seamlessly to varied underlying data sources. Model Context Protocol (MCP) Integration Agent leverages the Model Context Protocol as a tool gateway, enabling structured resource access and secure external API/resource management. Insight Analysis & Narrative Generation Agent provide contextual explanations and data storytelling, potentially leveraging public/private LLMs. 

The Semantic Layer ensures business meaning and logic are consistently applied, translating user queries and agent requests into context-aware SQL/statements. It supports modular updates and versioning, promoting business logic consistency across the platform. 

Vizro framework provides AI-native, configuration-driven tools that can be called using modern A2A and MCP protocols by AI agents for development of enterprise grade dashboards and visualizations from data which can be stored in a variety of formats in the data source such as pandas DataFrames, SQL, JSON etc.  

Data Sources & Pipeline Layer supports connection to databases, warehouses, lakes, and APIs. Event streaming infrastructure (Kafka, Flink) manages real-time analytics, feeding both anomaly detection and dashboard visualizations.

The system's architecture is a modular, multi-layered framework designed to process natural language user queries, retrieve data, and present visualizations and insights. The architecture is organized into four primary layers: the UI Layer, the Guardrail Layer, the central Orchestrator, and the Multi Agent Layer.

2. ## **2.2. Architecture Subsystems** {#2.2.-architecture-subsystems}

As portrayed in Figure 1, Agentic BI is proposed to consist of multiple sub-services accounting for separate features of the systems. These subsystems would collaborate with each other to effectively process incoming requests from users. Below is a detailed breakdown of all the subsystems proposed to be used in Agentic BI.

1. **User Interface Layer**  
   The User Interface (UI) Layer works as the entrypoint for users interacting with Agentic BI. It mainly consists of two components- the Chat Interface and the Dashboard. Figure 2 shows the proposed mock to the UI setup that would be used in the Agentic BI system.

* **Chat Interface:** The chat section is the primary source through which a user would input their natural language queries to interact with the agentic backend. Here, the user would describe the plots they want  to add or the problem they are facing and ask for a data-backed visualization to deep dive into the same business problem. The user would expect the agentic backend to return plot(s) through the dashboard component of the UI layer.  
* **Dashboard:** This section would serve as the display area in which the final outputs of the agentic backend will be rendered. The plots must be interactive and descriptive in nature.

**![][image2]**

Fig. 2-  Mock User Interface for Agentic BI

2. **Guardrail Layer**  
   The Guardrail Layer is placed between the Chat Interface and the core system logic, maintaining security and governance. It is responsible for performing checks on input user queries and outgoing plots, enforcing privacy and business policies. It is required to identify and flag any incoming threats and attacks, ensuring the durability and security of customer data, while ensuring full policy compliance.

3. **Orchestrator**  
   The orchestrator acts as the ‘manager’ for the specialized agents in the multi-agent Agentic BI system. It will leverage technologies like CrewAI, LangGraph and Agent-to-Agent (A2A) protocol for handling task distribution among the agents. It receives the sanitized query from the user, identifies tasks that need to be performed and delegates them among the available specialized agents. It is also responsible for routing information between the agents, ensuring the right piece of context goes to the right agent. 

   

   

4. **Database Layer**  
   The database layer is not in-house for the Agentic BI system. It would be hosted on the customer's side through their warehousing technologies. Agentic BI would require the users to connect the system with their data through a relational database structure. In this project, PostgreSQL databases hosted through a docker container would be used for this purpose. 

   Another data section required for the functioning is a vector database (VectorDB) which will be generated by the Data ingestion agent and required by other agents for their execution. This would be stored in-house in the Agentic BI system. 

5. **Data Ingestion Agent**  
   This the the first of many agents in the multi-agents system leveraged by Agentic BI. It is responsible for onboarding a user’s database into the system. It directly connects with the docker container of the user’s data, parses metadata and extracts structural information like table definitions, primary keys,joins, etc. and converts all the gathered information in a vector database. It ensures that the schema description is LLM-friendly and provides a semantic view of the user’s underlying database, so that the agentic subsystem can efficiently interact with the data to form queries and plots. 

**![][image3]**

Fig. 3- Sequence Diagram of Data Ingestion and SQL Generation

6. **NL-to-SQL Agent**  
   The NL-to-SQL agent, along with the Visualization Agent is the most important individual agent of this system, acting as the central translation engine, turning the user’s natural language problem into an executable SQL query that can be used by subsequent agents to interact with the underlying database. It takes as input the natural language intent of the user, the context of the dashboard, and the semantic view from the Ingestion Agent, and produces executable database queries. The agent outputs a Candidate SQL query along with a "tile intent," which summarizes the analytics intent-e.g., measures, dimensions, and time windows of interest. 

7. **SQL Execution Agent**  
   The SQL Execution Agent is an operational runtime for data retrieval. It executes the candidate SQL against the target data warehouse and manages the retrieval process. The key features of this agent include a Validation and Retry Loop wherein-if the database returns syntax errors, join inconsistencies, or filter mismatches-the agent automatically tries to correct the query and re-executes it until valid data is retrieved. If the execution is successful, the agent cleans, aggregates, and stages the results in a temporary tile store and assigns a unique identifier for the processing steps downstream.

![][image4]

Fig. 4- Sequence diagram of SQL Execution

8. **Visualization Agent**  
   The Visualization Agent is responsible for the presentation layer, transforming staged data into visual artifacts. Based on data type and granularity, it selects the most effective visualization format to communicate the results properly-for example, bar charts when comparing values, line charts to show trends. It creates accurate chart specifications using staged data that are compatible with frameworks like Plotly (Vizro). Equally, it generates a short interpretive summary-a text in natural language that describes the analytical outcome.

   

![][image5]

Fig. 5- Sequence diagram of Data Visualization

9. **Business Insights Agent**  
   This optional agent provides high-level analytical reasoning, scanning one or more generated dashboard tiles for deeper patterns not immediately obvious from the visual chart alone, such as anomalies, correlations, or emerging trends. The agent synthesizes these statistical findings into human-readable narratives that link up with visual tiles to provide contextual "data storytelling" capabilities to the end user.



3. # **Technology Descriptions** {#technology-descriptions}

   1. ## **3.1. Client Technologies** {#3.1.-client-technologies}

      1. ### **3.1.1. Vizro (Dash/Plotly Framework)** {#3.1.1.-vizro}

Vizro is a modern, open-source, AI-native dashboarding framework developed and maintained by McKinsey & Company. It is built on top of Plotly's Dash framework and React, providing a configuration-driven, low-code approach to building production-ready, enterprise-grade dashboards. Vizro was selected as the primary frontend framework for Agentic BI for several compelling reasons.

First, Vizro's design philosophy aligns directly with the project's goal: enabling the rapid construction of complex, interactive analytical dashboards without the overhead of full-stack web development. It provides a clean Python API that abstracts away the underlying Dash callbacks, ReactJS components, and CSS, enabling the team to focus on the intelligence layer rather than UI plumbing. Second, Vizro natively supports the Plotly chart library, meaning that the charts generated by the Visualization Agent can be passed directly to the dashboard's rendering pipeline without any format conversion. Third, Vizro's architecture is inherently compatible with agentic AI workflows. It supports integration with LangChain, CrewAI, and the Model Context Protocol (MCP), which positions it as a forward-looking choice for the evolving landscape of AI-driven analytics tools. The Vizro dark theme (`vizro_dark`) is set as the default Plotly template in the Visualization Agent, ensuring visual consistency across all rendered charts.

      2. ### **3.1.2. Web Application Execution Environment** {#3.1.2.-web-app}

The client application is served as a web application accessible via a standard web browser. The underlying execution environment is a Python WSGI server (provided by Dash's Flask server), which runs on port 8050 within a Docker container. This approach was chosen because it enables zero-installation access for end users: they simply navigate to the server's URL and interact with the full intelligence pipeline through a browser-based chat interface and a dynamic visualization dashboard. The containerized environment ensures that the application behaves identically regardless of the host operating system.

   2. ## **3.2. Middle-Tier Technologies** {#3.2.-middle-tier-technologies}

      1. ### **3.2.1. Orchestration Frameworks (LangGraph)** {#3.2.1.-orchestration}

LangGraph is a framework built on top of LangChain for creating stateful, multi-actor agentic applications. Unlike simple linear chains, LangGraph allows developers to define arbitrary computational graphs where each node is an agent or a processing step, and edges define the flow of control. Critically, LangGraph supports *conditional edges*, which enable dynamic routing based on the outcome of each node. This is the foundational mechanism behind the correction loops in Agentic BI.

The choice of LangGraph over simpler orchestration patterns is motivated by the inherent complexity of the BI pipeline. A successful end-to-end execution requires: (1) generating a valid SQL query, (2) executing it against an external database, (3) evaluating the results, and (4) generating a visualization — all of which can fail independently. LangGraph's `StateGraph` abstraction allows the system to maintain a shared `TypedDict` state object that flows through the entire pipeline. Each node reads from and writes to this shared state, enabling a clean, inspectable execution trace. The graph's conditional routing logic (`route_after_generate`, `route_after_check`, `route_after_visualize`) allows the system to gracefully handle failures, retry intelligently, and surface partial successes to the calling application. The Visualization Agent also implements its own *nested* `StateGraph` internally (with `decide` and `render` nodes), demonstrating a composable, layered architecture.

      2. ### **3.2.2. Cloud-Based Large Language Models (OpenRouter API)** {#3.2.2.-llm}

The reasoning and language understanding capabilities of the system are provided by large language models (LLMs) accessed through the OpenRouter API. OpenRouter is a unified LLM API gateway that provides access to a wide range of frontier models — including Llama, Mistral, GPT, and Claude — through a single, OpenAI-compatible endpoint. For Agentic BI, the default model is `meta-llama/llama-3.3-70b-instruct`, a state-of-the-art instruction-tuned open-weight model from Meta.

The decision to use a cloud-based LLM API rather than a solely locally-hosted model is driven by the demands of the task. Generating accurate, complex SQL queries across a rich, multi-schema database like AdventureWorks requires strong reasoning capabilities and a large effective context window. The Llama 3.3 70B model excels at these structured code generation tasks. Furthermore, the OpenRouter gateway provides model flexibility: because the codebase uses a `LLM_MODEL` environment variable and the standard `langchain_openai.ChatOpenAI` client, the underlying model can be swapped at any time without code changes — for example, switching to `openai/gpt-4o-mini` for lower latency or `google/gemini-pro-1.5` for a larger context window.

      3. ### **3.2.3. Embedding Models (HuggingFace all-MiniLM)** {#3.2.3.-embeddings}

Semantic search and context retrieval in the GraphRAG pipeline are powered by the `all-MiniLM-L6-v2` embedding model from the `sentence-transformers` library, loaded locally via the `langchain-huggingface` integration. This model produces 384-dimensional dense vector embeddings from text inputs and is optimized for semantic similarity tasks.

The embedding model was chosen for this role because it is compact (only ~23 million parameters), fast enough to run without GPU acceleration, and highly effective for document retrieval tasks. Since it runs locally, it introduces no API cost, no network latency, and no data privacy concerns — database schema information never leaves the host machine during the embedding phase. The same model is used both during the schema ingestion phase (to embed DDL documents into ChromaDB) and during the query phase (to embed the user's natural language question for similarity search), ensuring that the embedding space is consistent and retrieval is accurate.

   3. ## **3.3. Data-Tier Technologies** {#3.3.-data-tier-technologies}

      1. ### **3.3.1. Client Relational Database (Dynamic User Database via Docker)** {#3.3.1.-rdbms}

Agentic BI is designed as a BI platform, not a standalone application tied to a specific database. Its architecture anticipates that real-world users will connect their *own* relational databases to the platform. The mechanism for this connection, as implemented in the current system, is via a Docker-containerized PostgreSQL instance. The user provisions their database as a Docker container, and the system connects via the standard `psycopg2` driver using environment-variable-driven configuration (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).

For demonstration and testing purposes, the system ships with the Microsoft AdventureWorks database, a sample OLTP database used widely in the software industry for testing and training. This database contains multiple schemas (`Sales`, `HumanResources`, `Production`, `Purchasing`, `Person`) with dozens of interconnected tables and rich foreign-key relationships — making it an excellent stress test for the NL2SQL and GraphRAG components of the system.

      2. ### **3.3.2. Vector Database for Schema Context (ChromaDB)** {#3.3.2.-vectordb}

ChromaDB is an open-source, Python-native vector database optimized for embedding and retrieval workflows in AI applications. In Agentic BI, ChromaDB serves as the semantic index for database schema information. During the schema ingestion phase, the `SchemaIngestionAgent` generates DDL (Data Definition Language) statements for each table in the connected database, embeds them using the `all-MiniLM-L6-v2` model, and persists the resulting embeddings in a ChromaDB collection stored on the local filesystem under `src/agents/chroma_db_data`. This path is also mounted as a Docker volume, ensuring the index persists across container restarts.

At query time, the `NL2SQLAgent` uses ChromaDB's `similarity_search` method to retrieve the top-K most semantically relevant table definitions for the user's question. These retrieved DDL definitions form the "schema context" that is injected into the LLM prompt, enabling the model to generate SQL that is specific and valid for the user's database without having to process the entire schema every time.

      3. ### **3.3.3. Graph Topology Mapping (NetworkX)** {#3.3.3.-networkx}

NetworkX is a Python library for the creation, manipulation, and analysis of complex networks (graphs). In Agentic BI, NetworkX is used to represent the *relational topology* of the connected database as a directed graph. During schema ingestion, the `SchemaIngestionAgent` queries the database's `information_schema` to extract all foreign key relationships and uses them to construct a `networkx.DiGraph` where each node is a table and each directed edge represents a foreign key reference from one table to another.

This graph is serialized to a JSON file (`schema_graph.json`) alongside the vector index at ingestion time. At query time, the `NL2SQLAgent` loads this graph and performs a 2-hop graph traversal starting from the seed tables identified by ChromaDB's vector similarity search. This traversal discovers *related* tables that may not have appeared in the initial semantic search but are likely needed to answer the user's question (e.g., resolving an employee ID to a name requires joining through related tables). By combining vector similarity search with graph traversal, the system implements a **GraphRAG** (Graph-augmented Retrieval-Augmented Generation) architecture that dramatically improves the accuracy and completeness of the SQL generation context.

4. # **Project Design** {#project-design}

   1. ## **4.1. Client Design** {#4.1.-client-design}

      1. ### **4.1.1. Dashboard Layout and Reactivity Patterns** {#4.1.1.-dashboard-layout}

The Agentic BI client is built using Vizro's configuration-driven layout system. The dashboard is divided into two main regions: a left-side panel housing the chat interface, and a right-side (or inline) region where the generated Plotly figures are dynamically rendered. Vizro's `Page` and `Layout` constructs are used to define this two-panel structure, while Vizro-native components (such as `Graph`) handle the rendering of Plotly figures within the layout.

Reactivity in the dashboard is managed through Dash's callback system, which Vizro wraps in a slightly higher-level API. The critical reactivity pattern is event-driven: a user submitting a message to the chat interface triggers a Dash callback that invokes the `OrchestratorAgent`, waits for the result, and then updates a `dcc.Store` object with the returned Plotly figure JSON. A second callback listens for updates to this store and re-renders the `Graph` component, causing the new chart to appear in the dashboard without a full page reload.

      2. ### **4.1.2. Chat Interface State Management** {#4.1.2.-chat-state}

The chat interface is implemented as a custom Vizro/Dash component that maintains a conversation history in the browser. Internally, state (the list of chat messages and the currently loading state) is maintained in `dcc.Store` components, which provide client-side persistent storage within a Dash session. The `loading` state disables the chat input, send button, and all other interactive controls while the agentic pipeline is running, preventing duplicate submissions and giving the user a clear signal that the system is processing. This is surfaced in the UI as an "Agent running, controls disabled" indicator.

Each call from the chat interface to the backend passes the user's raw message string to the `OrchestratorAgent.run()` method. The method is invoked asynchronously within the Dash callback (using Dash's long callback pattern or a background execution approach), and the callback remains open until the result is returned, at which point the UI is updated with both the assistant's textual response (e.g., an error message or a success acknowledgment) and the rendered Plotly figure.

   2. ## **4.2. Middle-Tier Design** {#4.2.-middle-tier-design}

      1. ### **4.2.1. Master Orchestration StateGraph Architecture** {#4.2.1.-stategraph}

The core of the Agentic BI middle tier is a master LangGraph `StateGraph` compiled into the `OrchestratorAgent`. The state object is defined as an `OrchestratorState` TypedDict with the following key fields: `user_question`, `sql_query`, `correction_hint`, `sql_success`, `result_dict`, `markdown`, `viz_success`, `figure_json`, `chart_spec`, `error_message`, `retry_count`, and `stage`.

The graph consists of four nodes: `generate_sql`, `execute_sql`, `check_execution`, and `visualize`. The entry point is always `generate_sql`. Conditional edges implement two distinct correction loops:

- **Correction Loop 1 (SQL Error):** If `execute_sql` returns `sql_success=False`, the `check_execution` node increments the retry counter, formulates a `correction_hint` string containing the error message, and the routing function sends control back to `generate_sql`. The NL2SQL agent receives the hint and attempts to regenerate the query differently.
- **Correction Loop 2 (Empty Results):** If `execute_sql` returns `sql_success=True` but with an empty result set, the `check_execution` node formulates a different hint advising the NL2SQL agent to try different joins, filters, or column names, and again routes control back to `generate_sql`.

A maximum retry count of `MAX_RETRIES = 2` at the orchestrator level prevents infinite loops. The system also supports a "partial success" mode: if SQL execution succeeds but visualization fails, the pipeline still returns `success=True` to the caller, and includes `viz_failed=True` with the visualization error surfaced separately.

      2. ### **4.2.2. Prompt Design and GraphRAG Retrieval for NL2SQL** {#4.2.2.-prompt-design}

The NL2SQL Agent's prompt is carefully engineered to maximize the accuracy of generated SQL. The prompt provides the LLM with: (1) a clearly defined role ("You are an expert PostgreSQL Data Analyst"), (2) the schema context retrieved via GraphRAG, (3) the user's question, and (4) an optional correction block (present only on retry) containing the previous SQL query's error and instructions to generate a different query.

The prompt includes a set of hard constraints that are critical for correctness with the AdventureWorks database:
- All schema, table, and column names must be wrapped in double quotes to preserve case sensitivity.
- Only two-part identifiers (e.g., `"Sales"."SalesOrderHeader"`) are permitted — three-part identifiers are explicitly forbidden.
- The output must be raw SQL only — no markdown code fences or explanations.
- IDs should be resolved to human-readable names (e.g., `FirstName`, `LastName`) by joining the appropriate tables.

The **GraphRAG retrieval** for the prompt context operates as follows: the user's question is embedded using `all-MiniLM-L6-v2` and compared against the ChromaDB index via `similarity_search(k=8)`, returning the 8 most semantically relevant table DDL documents. The table names from these documents serve as "seed nodes" for a 2-hop traversal of the NetworkX foreign key graph. For each seed table, the traversal visits both successors (tables that the seed table references) and predecessors (tables that reference the seed table) up to 2 hops away. The DDL for all tables in the expanded set is then looked up from a pre-built dictionary (`schema_dict.json`) and concatenated to form the final schema context string for the LLM prompt.

      3. ### **4.2.3. Nested StateGraph for Visualization Validation** {#4.2.3.-viz-stategraph}

The Visualization Agent implements its own internal LangGraph `StateGraph`, separate from the master orchestrator's graph. This nested graph handles the chart generation task with its own internal retry logic. The state is defined by `VisualizationState`, which tracks: `user_question`, `df_columns`, `df_sample`, `df_row_count`, `chart_spec_json`, `chart_spec`, `figure_json`, `error_message`, and `retry_count`.

The graph has two nodes: `decide` (chart type selection) and `render` (chart rendering). The `decide` node calls the LLM with the user's question and a sample of the data (column names plus dtypes, and the first 5 rows in markdown format) and asks it to return a structured JSON `ChartSpec` object specifying the chart type and column mappings. This JSON is validated with Pydantic's `ChartSpec.model_validate()`. If parsing or validation fails, the node increments the retry counter and the conditional edge routes control back to `decide`. If successful, control passes to `render`, which deterministically maps the `ChartSpec` to a `vizro.plotly.express` function call and executes it. If rendering fails (e.g., because the LLM specified a column that doesn't exist in the DataFrame), control returns to `decide` for another attempt.

      4. ### **4.2.4. Guardrail Validation Rules** {#4.2.4.-guardrails}

The Guardrail Layer is a lightweight, stateless validation module that screens all incoming user prompts before they reach the orchestration pipeline. The guardrails are implemented as a set of compiled regular expression patterns that match common adversarial prompt injection techniques. Patterns include: attempts to override system instructions (e.g., "ignore previous instructions"), attempts to reveal the system prompt, attempts to switch the model into a "jailbreak" or "developer" mode, injection of low-level LLM instruction tokens (e.g., `[INST]`, `<<SYS>>`), and attempts to execute arbitrary shell commands or run destructive SQL statements directly via the prompt (e.g., `DROP TABLE`, `DELETE FROM ... WHERE 1=1`).

The design philosophy of the guardrails is intentionally permissive at the SQL level: they do not block valid SQL queries embedded in a user question, because the SQL generated by the NL2SQL agent is what matters. The guardrails focus exclusively on prompt-level abuse, ensuring the system cannot be used to exfiltrate the system's configuration, override agent behavior, or execute malicious commands through the conversational interface.

   3. ## **4.3. Data-Tier Design** {#4.3.-data-tier-design}

      1. ### **4.3.1. Database Schema Extraction and DDL Handling** {#4.3.1.-ddl}

The schema extraction process is designed to produce LLM-friendly, information-rich representations of the database structure. For each table, the `SchemaIngestionAgent` queries `information_schema.tables`, `information_schema.columns`, and `information_schema.table_constraints` / `information_schema.key_column_usage` / `information_schema.constraint_column_usage` to gather: the table name and schema, all column names with their data types and nullability constraints, and all foreign key relationships including the referenced table schema, table name, and column name.

This information is formatted into a synthetic `CREATE TABLE` DDL statement that also includes commented-out foreign key annotations (e.g., `-- ColumnName references Schema.Table(Column)`). This format was chosen because LLMs are trained extensively on SQL DDL and are highly effective at interpreting `CREATE TABLE` syntax. The foreign key comments provide explicit join hints without introducing invalid syntax.

      2. ### **4.3.2. Hybrid Vector and Graph Indexing Strategy** {#4.3.2.-hybrid-index}

The data-tier indexing strategy is hybrid: it combines dense vector embeddings (for semantic similarity) with a directed graph structure (for relational topology). The ChromaDB vector index enables efficient approximate nearest-neighbor search over the semantic representations of table names, column names, and data types. The NetworkX directed graph encodes the explicit foreign key relationships between tables, providing a structural complement to the semantic search.

The two indices are built simultaneously during the schema ingestion pipeline and stored in the same directory (`chroma_db_data`). The vector index is stored in ChromaDB's native persistence format, while the graph is serialized as a node-link JSON document (`schema_graph.json`) and the full DDL dictionary is stored as a flat JSON object (`schema_dict.json`). At query time, the `NL2SQLAgent` loads both the graph and the dictionary, uses the vector index to identify seed tables, and then uses the graph to expand context through 2-hop traversal.

5. # **Project Implementation** {#project-implementation}

   1. ## **5.1. Client Implementation** {#5.1.-client-implementation}

      1. ### **5.1.1. Building Custom Vizro Components** {#5.1.1.-vizro-components}

The Agentic BI dashboard required several customizations beyond Vizro's standard component library. The chat interface itself is not a native Vizro component, so it was built using raw Dash `html` and `dcc` components and wired into the Vizro `Page` layout. Specifically, the chat input box is a `dcc.Input` component; the send button is an `html.Button`; the conversation history is rendered via a `dcc.Markdown` component (updating dynamically with each exchange); and the loading overlay is implemented using the `dcc.Loading` wrapper applied to the output area.

A key customization is the **auto-horizontal-bar-chart** logic in the `build_figure` function. When the `VisualizationAgent` selects a `bar` chart type and the categorical X-axis has more than 10 unique values, the function automatically swaps the `x` and `y` arguments, sorts the DataFrame by the numeric column in ascending order, and passes the swapped arguments to `vizro.plotly.express.bar`. This results in a horizontal bar chart with bars sorted from smallest at the top to largest at the bottom — a UX pattern that is significantly more readable for dense categorical comparisons than a vertical bar chart with overlapping labels.

      2. ### **5.1.2. Implementing Asynchronous Callbacks** {#5.1.2.-async-callbacks}

The most significant implementation challenge on the client side is managing the asynchronous nature of agentic pipeline execution. LLM API calls can take 5–30 seconds, and database queries add additional latency. During this time, the UI must remain responsive: the user should see a clear "loading" state, and all input controls should be disabled to prevent double-submission.

This is achieved through Dash's callback architecture. A single Dash callback with `prevent_initial_call=True` is triggered by the "Send" button click event. This callback immediately renders the user's message in the chat history and begins executing the orchestrator pipeline synchronously within the callback context. Dash's `dcc.Loading` component automatically detects that the callback is running and displays a loading spinner over the output area. The callback returns only after the full orchestrator pipeline completes (successfully or with an error), at which point it updates the chat history with the assistant's response and, if a figure was produced, updates the `dcc.Store` that triggers the chart rendering callback.

   2. ## **5.2. Middle-Tier Implementation** {#5.2.-middle-tier-implementation}

      1. ### **5.2.1. Integrating OpenRouter API for Llama 3.3** {#5.2.1.-openrouter}

All LLM calls in the system are made via the `langchain_openai.ChatOpenAI` class, which implements the OpenAI API specification. OpenRouter provides an OpenAI-compatible endpoint, so integration required only two changes from a standard OpenAI setup: providing the `OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"` as the `base_url` parameter, and providing the `OPENROUTER_API_KEY` environment variable as the `api_key` parameter. The model is set via `LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")`, enabling model switching without code changes.

LangChain's LCEL (LangChain Expression Language) pipe operator (`|`) is used to compose prompt templates, the LLM client, and output parsers into clean, readable chains. For example, the NL2SQL chain is expressed as `chain = prompt | self.llm | StrOutputParser()`. For the Visualization Agent's chart-spec generation, `llm.with_structured_output(ChartSpec)` is not used; instead, the LLM is prompted to return raw JSON, which is then parsed and validated by Pydantic's `ChartSpec.model_validate()` with extensive fallback logic to handle common LLM JSON generation quirks.

      2. ### **5.2.2. Implementing the 2-Hop Schema Graph Traversal** {#5.2.2.-graph-traversal}

The 2-hop graph traversal is implemented in the `NL2SQLAgent.get_relevant_schema()` method. After ChromaDB returns the top-K similar table documents, their `table_name` metadata values are extracted as the initial `seed_tables` set. If the schema graph and dictionary are available (loaded from `schema_graph.json` and `schema_dict.json`), the traversal proceeds as follows:

A `final_tables` set is initialized with the seed tables. A `current_layer` set is also initialized with the seed tables. For each of the 2 hops, a `next_layer` set is populated by iterating over each node in `current_layer` and collecting all its successors (tables it references via foreign keys) and predecessors (tables that reference it via foreign keys) from the NetworkX graph. Any newly discovered tables (not already in `final_tables`) are added to both `final_tables` and `next_layer`, and `current_layer` is updated to `next_layer` for the next iteration. After 2 hops, the DDL for every table in `final_tables` is looked up from `schema_dict` and concatenated into the final context string. If the graph is unavailable (e.g., before ingestion), the system falls back to using only the raw ChromaDB document content.

      3. ### **5.2.3. Handling Multi-Stage Correction Loops** {#5.2.3.-correction-loops}

Two distinct correction loops operate at different levels of the pipeline, each addressing a different failure mode.

The **SQL Execution correction loop** operates within the `SQLExecutor` (which uses its own internal `StateGraph`). When `execute_sql_node` fails due to a database exception, the error message is passed to `fix_query_node`, which calls the LLM with the broken query and the Postgres error message. The LLM produces a corrected query, and the cycle repeats up to 3 times. As a preprocessing step before every execution attempt, `execute_sql_node` applies a deterministic regex-based quote-fixer (`_ensure_quoted_identifiers`) that automatically adds double quotes around schema names, table names, and CamelCase alias-qualified column names — catching the most common class of case-sensitivity errors without an LLM call.

The **Orchestrator-level correction loop** operates at a higher level and addresses two failure modes: SQL execution failure (already handled by the SQLExecutor, but also surfaced upward if all internal retries are exhausted) and empty result sets. When the `check_execution_node` detects that the SQL succeeded but returned 0 rows, it generates a hint specifically tailored to this failure mode ("The query returned 0 rows. Try different table joins, filters, or column names.") and routes control back to `generate_sql` with the hint. This two-level correction architecture allows the system to handle both low-level SQL syntax errors (handled by the SQLExecutor) and higher-level semantic errors (handled by the Orchestrator), providing robust error recovery across the full query-to-data pipeline.

   3. ## **5.3. Data-Tier Implementation** {#5.3.-data-tier-implementation}

      1. ### **5.3.1. Executing the Schema Ingestion Pipeline** {#5.3.1.-ingestion-pipeline}

The schema ingestion pipeline is executed as a standalone script (`schema_ingestion_agent.py`) by running `python schema_ingestion_agent.py` from the `src/agents` directory. It connects to the PostgreSQL database using `psycopg2.connect(**DB_CONFIG)`, queries the `information_schema` to discover tables in the `Person`, `Sales`, `Production`, `Purchasing`, and `HumanResources` schemas, and iterates through each table to build its DDL document.

Before building the new index, the ingestion script clears the existing contents of the `VECTOR_DB_PATH` directory (deleting all files and subdirectories) to prevent duplicate embeddings from accumulating across multiple ingestion runs. It then initializes `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` locally and calls `Chroma.from_documents()` to create and persist the new vector store. After the store is created, it serializes the NetworkX graph with `nx.node_link_data()` and writes it to `schema_graph.json`, and writes the DDL dictionary to `schema_dict.json`. A re-ingestion is required whenever the database schema changes (e.g., new tables are added or columns are modified).

      2. ### **5.3.2. Building the Hybrid GraphRAG Pipeline for Schema Retrieval** {#5.3.2.-graphrag-pipeline}

The GraphRAG schema retrieval pipeline is initialized once per `NL2SQLAgent` instance. The constructor loads the ChromaDB vector store from disk (using `Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=...)`) and deserializes the graph (`nx.node_link_graph(json.load(...))`) and dictionary from their JSON files. If either file is missing (e.g., the schema has not yet been ingested), the constructor logs a warning and sets both to `None`, causing `get_relevant_schema()` to fall back to returning the raw ChromaDB document content without graph traversal.

The complete retrieval flow is: (1) embed the user's question → (2) ChromaDB similarity search → (3) extract seed table names → (4) NetworkX 2-hop traversal → (5) DDL dictionary lookup → (6) concatenate context string → (7) inject into LLM prompt. The NL2SQLAgent also applies a `FullName` column auto-merge pre-processing step in the VisualizationAgent: if both `FirstName` and `LastName` columns are present in the SQL query results, a derived `FullName` column is added automatically before the data is passed to the chart-spec LLM call, improving the informativeness of visualizations involving people.

6. # **Testing and Verification** {#testing-and-verification}

   1. ## **6.1. Unit Testing Strategy** {#6.1.-unit-testing}

      1. ### **6.1.1. Component-Level Agent Tests** {#6.1.1.-component-tests}

The unit testing strategy for Agentic BI follows the principle of testing each agent in isolation, independent of the other agents and of the live database and LLM API. This is achieved through dependency injection and mocking.

For the **NL2SQLAgent**, unit tests verify the `get_relevant_schema()` method by pre-loading a test ChromaDB collection and a test `schema_graph.json` with known content, then asserting that the 2-hop traversal correctly expands the seed table set. The `generate_sql()` method is tested by mocking the `ChatOpenAI` client (via `unittest.mock.patch`) to return predetermined SQL strings, and then asserting that the method correctly injects the correction hint into the prompt when one is provided and strips markdown code fences from the response.

For the **SQLExecutor**, unit tests focus on the deterministic quote-fixer (`_ensure_quoted_identifiers`). A set of known-bad SQL strings (e.g., unquoted `HumanResources.Employee`, partially quoted schemas) are passed through the function, and the output is asserted against the expected fully-quoted version. The `fix_query_node` is tested by mocking `ChatOpenAI` and asserting that the LLM-generated fix is correctly cleaned (markdown stripped, SELECT statement extracted).

For the **VisualizationAgent**, unit tests verify the `build_figure()` function by instantiating it with a known `ChartSpec` and a test `pd.DataFrame`, and asserting that the correct `vizro.plotly.express` function is called with the correct arguments. The auto-horizontal-bar-chart logic is explicitly tested with a DataFrame that has more than 10 unique categorical values.

For the **Guardrails** module, unit tests verify all deny patterns: each DENY_PATTERN is tested with a known-malicious input string and the function is asserted to return `(False, "...")`, while benign data questions are asserted to return `(True, "")`.

   2. ## **6.2. Integration Testing** {#6.2.-integration-testing}

      1. ### **6.2.1. End-to-End Pipeline Execution** {#6.2.1.-e2e-pipeline}

Integration testing verifies the interactions between agents in the context of the complete pipeline. A key integration test is the **Orchestrator-NL2SQL-SQLExecutor integration**: the `OrchestratorAgent` is initialized with a real `NL2SQLAgent` and a real `SQLExecutor` (connected to the AdventureWorks test database) but with a mocked `VisualizationAgent`. A natural language query is submitted, and the test asserts that: (a) a valid SQL query is produced, (b) the SQL query executes successfully against the database, (c) the returned DataFrame is non-empty, and (d) the orchestrator returns `success=True` and `sql=<non-empty string>`.

A second integration test covers the **correction loop**: a query is crafted that is known to initially generate invalid SQL for the target schema (e.g., by using a non-existent column name that is common in other databases). The test asserts that after a correction loop, the orchestrator eventually returns a successful result, demonstrating that the LLM can leverage the error feedback to correct its query.

**Agent-to-LLM integration** tests verify that the `langchain_openai.ChatOpenAI` client correctly communicates with the OpenRouter API endpoint. These tests require a valid `OPENROUTER_API_KEY` environment variable and make real API calls. They verify that the model identifier is correctly passed, the response is a valid string, and the chain (prompt | llm | StrOutputParser) functions end-to-end.

   3. ## **6.3. End-to-End Database Scenario Testing** {#6.3.-e2e-db-tests}

      1. ### **6.3.1. Custom Workflows with AdventureWorks Sample DB** {#6.3.1.-adventureworks-tests}

The primary end-to-end (E2E) testing mechanism for Agentic BI is a set of natural language queries run against the AdventureWorks database. These queries were carefully designed to stress-test different aspects of the NL2SQL and GraphRAG pipeline, covering a range of SQL complexity levels:

1. **"Show me the top 10 best-selling products by total revenue."** Tests: `SUM()` aggregation, multi-table JOIN, `GROUP BY`, `ORDER BY DESC`, `LIMIT`. Expected Visualization: Bar Chart.
2. **"What is the total sales revenue by year and month?"** Tests: Date parsing/extraction, time-series grouping. Expected Visualization: Line Chart.
3. **"Show me the distribution of employees by marital status."** Tests: Simple `COUNT()` aggregation with single-column `GROUP BY`. Expected Visualization: Pie/Donut chart.
4. **"What is the average sick leave hours by job title?"** Tests: Categorical grouping with `AVG()`. Expected Visualization: Horizontal Bar Chart.
5. **"Show me total sales amount by sales territory country."** Tests: Cross-schema JOIN between sales and territory tables. Expected Visualization: Bar chart.
6. **"Which top 5 vendors do we spend the most money with?"** Tests: Purchasing schema JOIN (`PurchaseOrderHeader` and `Vendor`). Expected Visualization: Bar chart.
7. **"Show me the total inventory quantity grouped by product subcategory."** Tests: Multi-table JOIN across `ProductInventory`, `Product`, and `ProductSubcategory`. Expected Visualization: Horizontal Bar chart.
8. **"Compare the standard cost versus the list price for all products."** Tests: Retrieval of two continuous numeric variables without aggregation. Expected Visualization: Scatter Plot.
9. **"Show me the number of orders placed online versus in-store."** Tests: Boolean flag (`OnlineOrderFlag`) for grouping and counting. Expected Visualization: Pie chart or Bar chart.
10. **"Who are the top 5 sales representatives by total sales, and what were their total sales?"** Tests: Complex conditional JOINs spanning `Sales` and `Person` schemas with ID-to-name resolution. Expected Visualization: Bar chart.

Each of these queries is executed through the full `OrchestratorAgent.run()` pipeline, and the test asserts that `result['success'] == True` and `result['figure'] is not None`. For queries where specific values can be predicted from the AdventureWorks dataset, the tests also assert on properties of the returned DataFrame (e.g., number of rows, column names).

7. # **Performance and Benchmarks** {#performance-and-benchmarks}

   1. ## **7.1. Objective Setting for Agentic Systems** {#7.1.-objective-setting}

      1. ### **7.1.1. Defining Success Metrics for NL2SQL and Visual Generation** {#7.1.1.-success-metrics}

Evaluating an agentic BI system presents a different challenge from evaluating traditional software: there is no single, universally correct answer for most natural language queries. The primary success metrics for Agentic BI are therefore defined as follows:

- **SQL Execution Success Rate (SESR):** The percentage of user queries for which the system produces a SQL query that executes successfully against the database (i.e., without a Postgres exception) and returns at least one row of data. An SESR of ≥ 80% on the AdventureWorks E2E test suite is the target benchmark.
- **Visualization Success Rate (VSR):** The percentage of successful SQL executions for which the VisualizationAgent also successfully produces a valid Plotly figure. A VSR of ≥ 90% conditional on SQL success is the target.
- **End-to-End Success Rate (EESR):** The percentage of user queries for which the full pipeline (SQL + Visualization) completes without errors. This is the product of SESR and VSR.
- **Correction Loop Effectiveness:** The percentage of initially-failing queries that are recovered by at least one correction loop iteration. This metric demonstrates the value of the self-correcting architecture.

   2. ## **7.2. Latency and Throughput Analysis** {#7.2.-latency}

      1. ### **7.2.1. Graph Traversal vs. Vector Search Speeds** {#7.2.1.-graph-vs-vector}

The GraphRAG retrieval pipeline introduces a latency overhead compared to vanilla RAG (vector search only). The ChromaDB similarity search over the AdventureWorks schema (approximately 70 tables) is extremely fast, typically completing in under 100 milliseconds on modern hardware. The NetworkX 2-hop graph traversal adds a negligible overhead (microseconds to low milliseconds), as the graph is small and already loaded into memory. The dominant step in the retrieval pipeline is the DDL dictionary lookup and string concatenation, which is also sub-millisecond. Therefore, the total GraphRAG retrieval overhead compared to vanilla RAG is minimal, while the quality of the retrieved context is substantially higher — particularly for queries that require multi-table joins involving tables not directly referenced in the user's question.

      2. ### **7.2.2. Pipeline Orchestration Overhead Using Cloud LLMs** {#7.2.2.-llm-latency}

The primary source of latency in the Agentic BI pipeline is LLM API calls to OpenRouter. Based on typical performance of `meta-llama/llama-3.3-70b-instruct` on the OpenRouter platform, each LLM call has an expected latency of approximately 3–8 seconds for a prompt of the size used in Agentic BI (schema context of up to 5,000 tokens plus the user question). A full successful pipeline execution (no correction loops) involves two LLM calls: one in `generate_sql_node` and one in `decide_chart_node`. This puts the expected end-to-end latency for a successful, no-retry execution at approximately **6–16 seconds**. Each correction loop iteration adds one additional LLM call for SQL regeneration or SQL fixing, plus the database round-trip time.

Database execution latency depends on query complexity and the PostgreSQL instance configuration but is typically in the range of 100–500 milliseconds for the types of analytical queries handled by the system. Total end-to-end latency for the E2E test queries ranges from approximately 10 seconds (simple queries, no retries) to 30+ seconds (complex queries requiring one or two correction loop iterations).

   3. ## **7.3. Accuracy and Reliability** {#7.3.-accuracy}

      1. ### **7.3.1. SQL Generation Success Rates with GraphRAG** {#7.3.1.-sql-accuracy}

A key design hypothesis of Agentic BI is that GraphRAG-augmented context retrieval produces more accurate SQL generation than vanilla RAG (pure vector similarity search). To validate this hypothesis, the E2E test suite of 10 AdventureWorks queries is used as a benchmark. The success rate of the GraphRAG system is compared against a baseline version of the system where the 2-hop graph traversal is disabled (i.e., the seed tables from ChromaDB are used directly as the context without graph expansion).

For the 10 E2E test queries, which include several multi-table queries requiring cross-schema joins (e.g., queries 5, 6, 7, and 10), the GraphRAG context provides the LLM with the full join chain, reducing the probability of "hallucinated" table names or incorrect join conditions. Query 10 ("Who are the top 5 sales representatives...") is a particularly strong test, as it requires joining across `Sales.SalesPerson`, `Sales.SalesOrderHeader`, `Person.Person`, and `HumanResources.Employee` — tables that a pure vector search might not all retrieve in a single top-8 result set, but which are easily discovered via 2-hop traversal from the `Sales.SalesPerson` seed node.

8. # **Deployment, Operations, Maintenance** {#deployment}

   1. ## **8.1. Deployment Strategy** {#8.1.-deployment}

      1. ### **8.1.1. Containerization with Docker Compose** {#8.1.1.-docker}

Agentic BI is deployed as a multi-container application orchestrated by Docker Compose. The `docker-compose.yml` defines three services:

**`db` (PostgreSQL + AdventureWorks):** Runs the `chriseaton/adventureworks:postgres` Docker image, which bundles PostgreSQL with the AdventureWorks sample database pre-installed. The service exposes port 5432 on the host and uses a named Docker volume (`pgdata`) for persistent data storage. The `platform: linux/amd64` flag ensures compatibility with Apple Silicon (M1/M2/M3) Macs running the image under Rosetta 2 emulation.

**`pgadmin` (Database GUI):** Runs the `dpage/pgadmin4` image, providing a web-based GUI for inspecting the database contents. Accessible at `http://localhost:8080` with the default credentials configured in the Compose file. This service is optional and is primarily useful for developers who want to verify schema ingestion results or inspect query execution plans.

**`app` (Agentic BI Application):** Built from the project's local `Dockerfile`, this container runs the Vizro dashboard and all agent code on port 8050. The service is configured with environment variables for the database connection (`DB_HOST=db`, `DB_PORT=5432`, etc.) and loads the `OPENROUTER_API_KEY` from an `.env` file via `env_file`. The `extra_hosts: host.docker.internal:host-gateway` entry allows the container to reach services running on the Docker host — originally designed for a locally-running Ollama instance. The `chroma_data` named volume persists the ChromaDB vector index and schema graph files across container restarts, so ingestion does not need to be re-run unless the database schema changes.

      2. ### **8.1.2. Integration with 3rd-Party LLM APIs (OpenRouter)** {#8.1.2.-api-integration}

For the project demo and production use, Agentic BI uses the OpenRouter API for LLM inference rather than a locally-running model. The API key is provided as the `OPENROUTER_API_KEY` environment variable, loaded from the `.env` file in the project root. The `app` Docker service receives this key via the `env_file` Compose directive, making it available to all agent code at runtime.

The LLM model can be changed at any time by setting the `LLM_MODEL` environment variable to any valid OpenRouter model identifier (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`, `meta-llama/llama-3.1-8b-instruct`). This flexibility is valuable for cost management: a cheaper, faster model can be used for development and testing, while a higher-capability model is used for the demo.

   2. ## **8.2. Operational Maintenance** {#8.2.-maintenance}

      1. ### **8.2.1. Client Database Onboarding** {#8.2.1.-db-onboarding}

Adding a new client database to the Agentic BI platform requires two steps: (1) updating the environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) in the `.env` file to point to the new database, and (2) re-running the schema ingestion script (`python schema_ingestion_agent.py`) to rebuild the ChromaDB vector index and the NetworkX schema graph for the new database. The `SchemaIngestionAgent` handles the schema discovery and indexing automatically; the user only needs to ensure that the new database is accessible from the application container and that the database user has `SELECT` permissions on the `information_schema`.

For a SaaS deployment, this onboarding process would be triggered automatically via an API endpoint, and the schema indices would be stored per-tenant rather than in a shared filesystem volume.

      2. ### **8.2.2. Managing API Keys and Operational Costs** {#8.2.2.-api-keys}

The `OPENROUTER_API_KEY` is the primary operational cost driver of the system. Each LLM call consumes tokens, and costs scale with prompt length (schema context size) and model selection. For cost management, operators can: (1) switch to a smaller, cheaper model via the `LLM_MODEL` environment variable, (2) reduce the `k` parameter in `similarity_search(k=8)` to reduce the number of tables in the schema context (though this may reduce SQL accuracy), or (3) implement caching of LLM responses for common queries.

9. # **Summary, Conclusions, and Recommendations** {#summary-conclusions}

   1. ## **9.1. Summary** {#9.1.-summary}

Agentic BI is a multi-agent AI system that automates the business intelligence pipeline — from a user's natural language question to an interactive data visualization — by orchestrating a set of specialized AI agents: a Natural Language to SQL agent with GraphRAG-augmented context retrieval, a SQL Execution agent with LLM-powered self-correction, a Visualization agent with a nested self-correcting StateGraph, and a Guardrail agent for prompt safety. The system demonstrates that a production-quality, end-to-end automated BI workflow is achievable using contemporary open-source tools (LangGraph, ChromaDB, NetworkX, Vizro, HuggingFace Transformers) combined with a cloud-hosted frontier LLM (Llama 3.3 via OpenRouter).

   2. ## **9.2. Conclusions** {#9.2.-conclusions}

The implementation of Agentic BI validates several key hypotheses: (1) GraphRAG-augmented context retrieval substantially improves NL-to-SQL accuracy over vanilla RAG for multi-table analytical queries; (2) nested StateGraph architectures enable composable, self-correcting agentic behaviors that are easier to reason about and maintain than monolithic chains; (3) deterministic preprocessing (the quote-fixer) and LLM-based correction (the fix query node) are complementary strategies that together address the full spectrum of SQL generation failure modes; and (4) the modular, layered architecture of the system makes it straightforward to swap individual components (e.g., the LLM model, the database, the visualization framework) without disrupting the pipeline.

   3. ## **9.3. Recommendations for Further Research** {#9.3.-recommendations}

Future work on the Agentic BI platform should investigate the following directions: (1) **Business Insights Agent** — implementing the planned agent that performs automated pattern detection, anomaly identification, and natural language narration over the generated visualizations; (2) **Multi-turn conversational context** — extending the pipeline to maintain session context across multiple user questions, enabling follow-up queries that refine or pivot from previous results; (3) **Support for non-relational data sources** — extending the platform to support graph databases, document stores, and columnar warehouses; and (4) **Benchmarking against NL2SQL leaderboards** — evaluating the GraphRAG NL2SQL component against standard benchmarks such as Spider and WikiSQL to quantify the accuracy improvement over baseline approaches.

# **Glossary**

| Term | Definition |
| :---- | :---- |
| Agentic | Pertaining to a system composed of autonomous agents capable of self-managed, goal-oriented actions, decision-making, and communication to complete complex analytical workflows. |
| LLM | Large Language Model. The core reasoning engine used by the agents to understand natural language, generate code (like SQL), and provide contextual analysis and narrative insights. |
| NL2SQL | Natural Language to SQL. The process or capability of translating a user's natural language question into an executable SQL query against the underlying database. |
| Multi-Agent Architecture | A system design pattern where multiple specialized and autonomous AI agents are orchestrated to collaborate, perform distinct responsibilities, and solve a complex analytical problem. |
| Anomaly Detection | A capability of the Monitoring Agent to automatically identify rare events, sudden shifts, or deviations in real-time data streams that do not conform to expected patterns. |
| Tile Intent | The structured output from the NL-to-SQL Agent that summarizes the user's analytical goal (e.g., measures, dimensions, time range) alongside the generated SQL, guiding the Visualization Agent. |
| Orchestrator | The central 'manager' leveraging CrewAI that receives sanitized queries, identifies required tasks, delegates them to specialized agents, and manages the flow of information and collaboration. |
| Guardrail Layer | A security and governance component positioned between the UI and the core system that validates inputs and filters outputs, enforcing privacy policies, access control, and compliance. |
| Semantic Layer | A conceptual model that sits above the raw database structure, ensuring that business logic, metrics, and data relationships are consistently applied and understood by all agents. |
| Vector Database (VectorDB) | A specialized database that stores the LLM-friendly schema descriptions and metadata as high-dimensional embeddings for efficient context retrieval by other agents. |
| CrewAI | The specific orchestration framework used to define, manage, and coordinate the roles, responsibilities, and communication protocols for all specialized agents. |
| Vizro | The open-source, AI-native visualization framework leveraged by the Visualization Agent for creating interactive, production-ready enterprise-grade dashboards. |
| A2A Protocol | Agent-to-Agent Protocol. An open communication standard used for structured, direct, and efficient exchange of information and context between autonomous agents. |
| MCP | Model Context Protocol. An emerging open standard that facilitates structured tool access and secure external API/resource management. |
| SGIS | Schema-Graph with Integrated Syntax. A modern approach used by the NL-to-SQL Agent to integrate schema relationships into the query generation process, improving semantic correctness. |
| LEVA | LLM-Enhanced Visual Analytics. A framework utilized to empower agents to perform interactive data exploration and dynamically adjust visualization analysis based on user intent. |

# **References**

**\[1\]** Microsoft, “The foundation for powering AI-driven operations: Fabric real-time intelligence | Microsoft Fabric Blog | Microsoft Fabric,” *Microsoft.com*, 2025\. \[Online\]. Available: [https://blog.fabric.microsoft.com/en-us/blog/the-foundation-for-powering-ai-driven-operations-fabric-real-time-intelligence/](https://blog.fabric.microsoft.com/en-us/blog/the-foundation-for-powering-ai-driven-operations-fabric-real-time-intelligence/). Accessed: Sep. 27, 2025\.

**\[2\]** McKinsey & Company, “The state of AI: How organizations are rewiring to capture value,” *McKinsey & Company*, Mar. 12, 2025\. \[Online\]. Available: [https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai).

**\[3\]** JulCsc, “See what’s new with the latest Power BI update \- Power BI,” *Microsoft.com*, Aug. 12, 2025\. \[Online\]. Available: [https://learn.microsoft.com/en-us/power-bi/fundamentals/desktop-latest-update](https://learn.microsoft.com/en-us/power-bi/fundamentals/desktop-latest-update).

**\[4\]** Tableau, “Tableau Research | Tableau Blog,” *Tableau.com*. \[Online\]. Available: [https://www.tableau.com/blog/research](https://www.tableau.com/blog/research).

**\[5\]** J. Jiang et al., "SiriusBI: A Comprehensive LLM-Powered Solution for Data Analytics in Business Intelligence," Proc. VLDB Endow., vol. 18, no. 12, pp. 4860–4873, 2025, doi: 10.14778/3750601.3750610.

**\[6\]** M. L. Bernardi, A. Casciani, and M. Cimitile, "Conversing with business process-aware large language models: the BPLLM framework," J. Intell. Inf. Syst., vol. 62, pp. 1607–1629, 2024, doi: 10.1007/s10844-024-00898-1.

**\[7\]** L. Lawrence and J. Butler, "A case study of large language models' effectiveness in diverse business applications: Developing a universal integration framework," The Pinnacle: A Journal by Scholar-Practitioners, vol. 2, no. 1, 2024, doi: 10.61643/c38193.

**\[8\]** Z. Zhan, E. Haihong, and M. Song, "Leveraging Large Language Model for Enhanced Text-to-SQL Parsing," IEEE Access, vol. 13, pp. 30497–30504, 2025, doi: 10.1109/ACCESS.2025.3540072.

**\[9\]** Z. Hong et al., "Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL," IEEE Trans. Knowl. Data Eng., 2025, doi: 10.1109/TKDE.2025.3609486.

**\[10\]** X. Liu et al., "A Survey of Text-to-SQL in the Era of LLMs: Where Are We, and Where Are We Going?," IEEE Trans. Knowl. Data Eng., vol. 37, no. 10, pp. 5735–5754, Oct. 2025, doi: 10.1109/TKDE.2025.3592032.

**\[11\]** A. Chopra and R. Azam, "Enhancing Natural Language Query to SQL Query Generation Through Classification-Based Table Selection," in Engineering Applications of Neural Networks (EANN 2024), L. Iliadis, I. Maglogiannis, A. Papaleonidas, E. Pimenidis, and C. Jayne, Eds., Cham: Springer, 2024, vol. 2141\.

**\[12\]** W. Zhao et al., "Enhancing Interaction Graph of Data Schema and Syntactic Structure with Pre-trained Language Model for Text-to-SQL," in Big Data, vol. 2301, L. Kong et al., Eds., Springer, 2025, pp. 159–173.

**\[13\]** S. Chu and J. Liu, "Based on BERT-GPT-GNN converged architecture: intelligent generation engine for complex SQL queries in business intelligence," Discover Artif. Intell., vol. 5, no. 1, Art. no. 147, 2025, doi: 10.1007/s44163-025-00381-y.

**\[14\]** Y. Wu et al., "Automated Data Visualization from Natural Language via Large Language Models: An Exploratory Study," Proc. ACM Manag. Data, vol. 2, no. 3, Art. no. 115, 2024, doi: 10.1145/3654992.

**\[15\]** Y. Zhao et al., "LEVA: Using Large Language Models to Enhance Visual Analytics," IEEE Trans. Vis. Comput. Graph., vol. 31, no. 3, pp. 1830–1847, 2025, doi: 10.1109/TVCG.2024.3368060.

**\[16\]** V. Dhanoa, A. Wolter, G. M. Leon, H.-J. Schulz, and N. Elmqvist, "Agentic Visualization: Extracting Agent-based Design Patterns from Visualization Systems," IEEE Comput. Graph. Appl., pp. 1–13, 2025, doi: 10.1109/MCG.2025.3607741.

**\[17\]** A. Bandi, B. Kongari, R. Naguru, S. Pasnoor, and S. V. Vilipala, "The Rise of Agentic AI: A Review of Definitions, Frameworks, Architectures, Applications, Evaluation Metrics, and Challenges," Future Internet, vol. 17, no. 9, p. 404, 2025, doi: 10.3390/fi17090404.

**\[18\]** A. Petrenko, "Agent-based approach to implementing artificial intelligence (AI) in service-oriented architecture (SOA)," Syst. Res. Inf. Technol., no. 1, pp. 104–123, 2025, doi: 10.20535/SRIT.2308-8893.2025.1.08.

**\[19\]** M. M. Karim et al., "Transforming Data Annotation with AI Agents: A Review of Architectures, Reasoning, Applications, and Impact," Future Internet, vol. 17, no. 8, p. 353, 2025, doi: 10.3390/fi17080353.

# **Appendices**

1. 

The PoC implementation of the project is available at [https://github.com/Shivang-Patel/AgenticBI](https://github.com/Shivang-Patel/AgenticBI) . The repository consists of:

* /src/agents \- Dataset stored in PostgreSQL Docker container and source codes for NL2SQL agent,  Schema Ingestion agent, SQL Execution agent.  
* /tests \- Test scripts for agents

