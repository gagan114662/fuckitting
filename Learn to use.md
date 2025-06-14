Learn to use
QuantConnect
and Explore
Features
CLOUD PLATFORM
Quant Research
in the Cloud
Quickly deploy quantitative
strategies on the QuantConnect
Cloud Platform.
Table of Content
1 Welcome
2 Getting Started
3 Security and IP
4 Organizations
4.1 Getting Started
4.2 Tier Features
4.3 Resources
4.4 Object Store
4.5 Support
4.6 Members
4.7 Administration
4.8 Billing
4.9 Credit
4.10 Training
5 Learning Center
5.1 Training
5.2 Educators
5.3 Course Structure
6 Projects
6.1 Getting Started
6.2 Structure
6.3 Files
6.4 IDE
6.5 Encryption
6.6 Debugging
6.7 Collaboration
6.8 Code Sessions
6.9 Shared Libraries
6.10 Package Environments
6.11 LEAN Engine Versions
7 Research
7.1 Getting Started
7.2 Deployment
8 Backtesting
8.1 Getting Started
8.2 Research Guide
8.3 Deployment
8.4 Results
8.4.1 Portfolio Margin Plots
8.5 Debugging
8.6 Report
8.7 Engine Performance
9 Datasets
9.1 Navigating the Market
9.2 Categories
9.3 Data Issues
9.4 Misconceptions
9.5 Licensing
9.6 Vendors
9.7 QuantConnect
9.7.1 US Equities
9.7.2 Crypto
9.7.3 Crypto Futures
9.7.4 CFD
9.7.5 Forex
9.7.6 Futures
9.7.7 Alternative Data
9.8 Alpaca
9.9 Charles Schwab
9.10 Interactive Brokers
9.11 Polygon
9.12 Samco
9.13 TradeStation
9.14 Tradier
9.15 Zerodha
10 Live Trading
10.1 Getting Started
10.2 Brokerages
10.2.1 QuantConnect Paper Trading
10.2.2 Interactive Brokers
10.2.3 TradeStation
10.2.4 Alpaca
10.2.5 Charles Schwab
10.2.6 Binance
10.2.7 ByBit
10.2.8 Tradier
10.2.9 Kraken
10.2.10 Coinbase
10.2.11 Bitfinex
10.2.12 Bloomberg EMSX
10.2.13 Trading Technologies
10.2.14 Wolverine
10.2.15 FIX Connections
10.2.16 CFD and FOREX Brokerages
10.2.17 Unsupported Brokerages
10.3 Deployment
10.4 Notifications
10.5 Results
10.6 Algorithm Control
10.7 Reconciliation
10.8 Risks
11 Optimization
11.1 Getting Started
11.2 Parameters
11.3 Objectives
11.4 Strategies
11.5 Deployment
11.6 Results
12 Object Store
13 Community
13.1 Code of Conduct
13.2 Forum
13.3 Discord
13.4 Profile
13.5 Quant League
13.6 Quant League Pro
13.7 Academic Grants
13.8 Integration Partners
13.9 Affiliates
13.10 Research
14 API Reference
14.1 Authentication
14.2 Project Management
14.2.1 Create Project
14.2.2 Read Project
14.2.3 Update Project
14.2.4 Delete Project
14.2.5 Collaboration
14.2.5.1 Create Project Collaborator
14.2.5.2 Read Project Collaborators
14.2.6 Nodes
14.2.6.1 Read Project Nodes
14.2.6.2 Update Project Nodes
14.3 File Management
14.3.1 Create File
14.3.2 Read File
14.3.3 Update File
14.3.4 Delete File
14.4 Compiling Code
14.4.1 Create Compilation Job
14.4.2 Read Compilation Result
14.5 Backtest Management
14.5.1 Create Backtest
14.5.2 Read Backtest
14.5.2.1 Backtest Statistics
14.5.2.2 Charts
14.5.2.3 Orders
14.5.2.4 Insights
14.5.3 Update Backtest
14.5.4 Delete Backtest
14.5.5 List Backtests
14.6 Live Management
14.6.1 Create Live Algorithm
14.6.2 Read Live Algorithm
14.6.2.1 Live Algorithm Statistics
14.6.2.2 Charts
14.6.2.3 Portfolio State
14.6.2.4 Orders
14.6.2.5 Insights
14.6.2.6 Logs
14.6.3 Update Live Algorithm
14.6.3.1 Liquidate Live Portfolio
14.6.3.2 Stop Live Algorithm
14.6.4 List Live Algorithms
14.6.5 Live Commands
14.6.5.1 Create Live Command
14.6.5.2 Broadcast Live Command
14.7 Optimization Management
14.7.1 Create Optimization
14.7.2 Update Optimization
14.7.3 Read Optimization
14.7.4 Delete Optimization
14.7.5 Abort Optimization
14.7.6 List Optimization
14.7.7 Estimate Optimization Cost
14.8 Object Store Management
14.8.1 Upload Object Store Files
14.8.2 Get Object Store Metadata
14.8.3 Get Object Store File
14.8.4 Delete Object Store File
14.8.5 List Object Store Files
14.9 Reports
14.9.1 Backtest Report
14.10 Account
14.11 Lean Version
14.12 Examples
Welcome
Welcome
QuantConnect is an open-source, community-driven algorithmic trading platform. Our trading engine is powered by LEAN , a
cross-platform, multi-asset technology that brings cutting-edge finance to the open-source community. We support Python and
C# programming languages.
Our Mission
Quantitative trading infrastructure should be open-source. Millions of financial engineers rewrite the same infrastructure and
then keep their work closed to make it harder for competing trading firms. We take a radically open approach to quant finance
and let our users focus on alpha, not infrastructure.
The future of finance is automated and we are the open-source infrastructure to power this future. Firms choose our opensource platform as it provides a 10-100x improvement in time to market and substantially reduces the risk of developing your
quantitative tools. As our brokerage and data integrations expand, this leverage will be more exceptional.
Everyone should have access to quantitative finance. Algorithmic trading is a powerful tool and we want to open it up to all
investors. We do this with transparent, scalable pricing that allows all investors to access quantitative finance. For more
information on our mission, check out our Manifesto .
Data Library
We provide an enormous library of data for your backtesting, research, and live trading. The library includes data for Equities,
Options, Futures, CFDs, Forex, Crypto, Indices, and alternative data. The library is roughly 400TB in size, contains trade data
that spans decades into the past, and comes in tick to daily resolutions. View the Dataset Market to see all of the datasets that
we have available, including their respective start dates, end dates, and resolutions. The following image shows the integrated
data providers:
Business Model
QuantConnect provides cloud infrastructure as a service, similar to many cloud compute vendors. We encourage quants and
start up firms to grow within our ecosystem to keep our pricing accessible to individuals and small firms.
For companies interested in running QuantConnect on-premise, we can install it within your corporate firewall and help you get
set up with financial data. We charge a set up and maintenance fee for these installations.
Getting Started
Getting Started
GET STARTED WITH QUANTCONNECT
Guide through creating a project, running your first backtest, and live algo trading.
Follow these steps to create, backtest, and paper trade a new algorithm:
1. Log in to the Algorithm Lab .
2. On the Projects page, click Create New Algorithm .
3. A new project opens in the Cloud Integrated Development Environment (IDE) .
4. At the top of the IDE, click the Build icon.
5. At the top of the IDE, click the Backtest icon.
The backtest results page displays your algorithmʼs performance over the backtest period.
6. At the top of the IDE, click the Deploy Live icon.
7. On the Deploy Live page, click the Brokerage field and then click Paper Trading from the drop-down menu.
8. Click Deploy .
The live results page displays your algorithmʼs live trading performance.
To deploy a live algorithm with a different brokerage, see the Deploy Live Algorithms section of the brokerage integration
documentation .
Security and IP
Security and IP
Introduction
You own all your intellectual property and your code. Your code is private by default unless you explicitly share it with the
community in the forums or with another team member via collaboration. You are creating valuable intellectual property, and we
respect this and wish to make it easier. We limit the QuantConnect staff members who have access to the database. If we ever
need access to your algorithm for debugging, then we will explicitly request permission first.
Intellectual Property
The following sections explain our commitment to protecting your intellectual property.
Ownership
You own all your intellectual property and your code. Your code is private by default unless you explicitly share it with the
community in the forums or with another team member via collaboration. You are creating valuable intellectual property; we
respect this and wish to make it easier. We document this publicly in our Terms of Service .
Alignment
Beyond words and our legal terms of service, we live this through the alignment of our business model with you, our clients. We
aim for you to become more successful, thereby growing within our ecosystem. We've served start-up quant funds as they've
grown from 0to1B+ AUM.
Reputation and Track Record
Established in 2012, we have a pristine reputation and a 10+ year record of protecting our community's intellectual property. We
have served more than 200,000 clients. If we were to violate the trust of even a single client, we'd lose the entire community's
faith - it is simply not worth it. We seek to change the future of finance and are driven by this mission.
Sharing
QuantConnect provides support with a specific support agent, not an anonymous team. When submitting a support ticket, you
explicitly grant that team member access to your project. You can remove the support agent from your project collaboration at
any time.
Security
We take a multi-level approach to security, from physical security to digital and information systems, internal processes, and
testing.
Physical Security of Servers
Physical access to our servers is limited to a few dedicated team members whom QuantConnect has vetted. Only those
credentialed team members can access the physical servers, and we schedule all work in advance. Work on the servers is
always done in pairs to prevent single rogue actors from accessing the servers. We host our servers in a world-class security
facility (Equinix) with security staff 24/7.
Information and Digital Security
We use all good common sense information security processes: passwordless servers, encryption in the database and backups,
encrypted traffic, and network monitoring. We keep most of our servers off the internet and only available on private networks
for the smallest possible surface area. We have regular network and code penetration testing. All code is containerized and
isolated in services so that root network access would provide little-to-no benefit.
Beyond these basics, we've built active monitoring technology that proactively detects and blocks threats. We have human
detection services to reduce the chances of brute-force attacks. We have documented processes for client notifications in the
event of strange network activity.
Processes
Deployment environments are automated and enforce code peer-review to be deployed, reducing the chances of a rogue
internal agent.
We limit staff access to the physical servers, restricting core database access to only a handful of senior staff. Database
credentials are carefully restricted in scope, access locations and frequently rotated.
Privacy
Protecting users' private and intellectual property is of utmost importantance to us. QuantConnect complies with GDPR and all
relevant privacy laws. We will never sell or publish your email address. We request knowledge of your real identity to ensure
compliance with our data licenses but accept using an alias on public profiles for privacy. For more information about what data
we collect, which tracking technologies we use, and how we use and share your data, see our Privacy Policy .
Code Encryption
When the above processes and track record are insufficient, we offer ways to encrypt your code while working on the
QuantConnect platform. Our encryption feature uses a locally hosted key to encrypt code at rest in our database with AES256
encryption. In case of a compromised database or rouge agent, your code would not be readable. For more information on this
feature, see Project Encryption .
On-Premise Installations
QuantConnect technology can be installed fully on-premise, fulfilling the requirements of even the strictest compliance
departments. This technology, called the Local Platform, provides the same user interface as our cloud environment with
support for fully offline "anonymous" project development. For more information on this feature, see the Local Platform .
Organizations
Organizations
An organization is a collection of members that share hardware resources, share access to datasets, and collaborate together to
develop projects. Hardware resources are used to run backtests, launch research notebooks, deploy live trading algorithms, and
store project data. You can create new organizations and join an number of exisiting existing ones. We offer several
organization tiers so you can tailor your team's subscriptions as you grow over time. For the times when you need access to a
QuantConnect engineer to help solve development issues, assign support seats among your team. The organization owner has
several special permissions, which includes defining the out-of-sample backtest period to avoid overfitting.
Getting Started
Learn the basics
Tier Features
Tiers to serve all team sizes
Resources
Share hardware nodes to research and trade
Object Store
File system that you can use in your algorithms to save, read, and delete data
Support
Connect with QC experts
Members
Effectively manage your team
Administration
Manage your organization
Billing
Configure your QC services
Credit
Gift to others or spend on QC services
Training
Get team members up to speed
See Also
Collaboration
Learning Center
Organizations > Getting Started
Organizations
Getting Started
Introduction
An organization is a collection of members that share hardware resources, share access to datasets, and collaborate together to
develop projects. Hardware resources are used to run backtests, launch research notebooks, deploy live trading algorithms, and
store project data. You can create new organizations and join existing ones. You can be a member in any number of
organizations. We offer several organization tiers so you can tailor your team's subscriptions as you grow over time. For the
times when you need access to a QuantConnect engineer to help solve development issues, assign support seats among your
team. There are several tiers of support seats to match the level of support your team requires.
Add Organizations
Follow these steps to add new organizations to your profile:
1. Log in to the Algorithm Lab.
2. In the top navigation bar, click Connected as: organizationName .
3. In the Switch Organization panel, click Create Organization .
4. Enter the organization name and then click Add .
The organization name must be unique. "Created Successfully" displays.
Switch Organizations
Follow these steps to switch organizations:
1. Log in to the Algorithm Lab.
2. In the top navigation bar, click Connected as: organizationName .
3. In the Switch Organization panel, click the name of the organization for which you want to connect.
The top navigation bar displays the new organization name.
Rename Organizations
Follow these steps to change your organization name:
1. Open the organization homepage .
2. Hover over the organization name and then click the pencil icon that appears.
3. Enter the new organization name and then click Save Changes .
"Organization Name Updated Successfully" displays.
Get Organization Id
To get the organization Id, open Organization > Home and check the URL. For example, the organization Id of
https://www.quantconnect.com/organization/5cad178b20a1d52567b534553413b691 is 5cad178b20a1d52567b534553413b691.
Out of Sample Period
To reduce the chance of overfitting, organization managers can enforce all backtests must end a certain number of months
before the current date. For example, if you set a one year out-of-sample period, the researchers on your team will not be able
to use the most recent year of data in their backtests. A out-of-sample period is helpful because it leaves you a period to test
your model after your done the development stage. Follow these steps to change the backtest out-of-sample period:
1. Open the organization homepage .
2. Scroll down to the Backtesting Out of Sample Period section.
3. Adjust the out-of-sample period duration or click on "No Holdout Period".
Organizations > Tier Features
Organizations
Tier Features
Introduction
An organization is a collection of members that share hardware resources, share access to datasets, and collaborate together to
develop projects. Hardware resources are used to run backtests, launch research notebooks, deploy live trading algorithms, and
store project data. You can create new organizations and join existing ones. You can be a member in any number of
organizations. We offer several organization tiers so you can tailor your team's subscriptions as you grow over time. For the
times when you need access to a QuantConnect engineer to help solve development issues, assign support seats among your
team. There are several tiers of support seats to match the level of support your team requires.
Organizations let you coordinate resources and teamwork on QuantConnect Cloud. There are 5 tiers of organizations and each
tier has its own set of features. Each account starts with a personal organization on the Free tier with access to one free
backtest node and one free research node. However, to accommodate the growth of your trading skills and business, you can
adjust the tier of your organization at any time. Higher tiers offer more live nodes to run more live algorithms, more backtesting
nodes for faster concurrent backtesting, and many other features.
Free Tier
The Free tier provides cloud access to datasets for all of the asset classes in our Datasets Market . The free data ranges from
minute to daily resolutions and can be used to either run backtests or perform analysis in the Research Environment . When
backtesting, Free organizations have access to our built-in auto-complete and debugging features in the web IDE. After a
successful backtest, Free organizations can use our report generator to create professional-grade reports that reflect their
backtest performance. Free organizations have access to our online documentation, community forum, YouTube channel, and
Learning Center.
Quant Researcher Tier
The Quant Researcher tier is designed for self-directed investors, students, academics, and independent traders seeking to
manage their personal portfolio. We recommend the Quant Researcher Pack to make the most of QuantConnect.
The Quant Researcher tier builds on the features included in the Free tier. Organizations on the Quant Researcher tier have
access to the QuantConnect API and can use the CLI to run Lean locally. When members in these organizations need assistance
from a QuantConnect engineer, support seats are available to request private support . Members within Quant Researcher
organizations that have the required permissions can adjust the resources within the organization.
In Quant Researcher organizations, members can use second and tick resolution data from the Datasets Market. There is no
limit on the number of projects these organizations can hold. They can produce up to 100KB of logs/backtest, 3MB of logs/day.
10 million orders/backtest, and can have up to two backtesting nodes to run up to two concurrent backtests. Members in these
organizations can have up to two active coding sessions in the organization. After a successful backtest, members in these
organizations can use parameter optimization tools to improve the performance of their backtest. When the members are ready
to deploy strategies live, Quant Researcher organizations can subscribe to up to 2 live trading nodes to unlock live trading with
real or paper money. Each live algorithm in a Quant Research organization can send up to 20 Telegram, Email, or Webhook
notifications per hour for free. SMS notifications and additional Telegram, Email, or Webhook notifications require QuantConnect
Credit (QCC).
Team Tier
The Team tier is designed for sophisticated individuals and teams of quant collaborators such as Quant Start Ups, Fintech
Companies, and Emerging Managers. We recommend the Team Pack to make the most of QuantConnect.
The Team tier expands on the features included in the Quant Researcher tier. Organizations on the Team tier can have up to 10
members and the members can collaborate on projects together. These organizations can produce 1MB of logs/backtest, 10MB
of logs/day, and there is no limit on the number of orders that they can place in backtests. Organizations on the Team tier can
have up to 10 backtesting nodes, 10 research nodes, and 10 live trading nodes. Members in these organizations can have up to
four active coding sessions in the organization.
To accommodate a large number of projects in Team organizations, these organizations can expand the capacity of their Object
Store up to 10GB. Annual contracts for onboarding services are available on request to get teams operational in the shortest
amount of time. When live trading, Team organizations have more options than the lower tiers because both the Trading
Technologies brokerage and our live Futures data provider are available. Each live algorithm in a Team organization can send up
to 60 Telegram, Email, or Webhook notifications per hour for free.
Trading Firm Tier
The Trading Firm tier is designed for growing quantitative firms, prop desks, hedge funds, ETF companies, professional teams
of quants, and sophisticated independent investors. It has special features for collaborating with consultants to protect the
investor IP. If you are a company on QuantConnect, we recommend the Trading Firm Pack to make the most of QuantConnect.
The Trading Firm tier builds on the features included in the Team tier. Organizations on the Trading Firm tier can have an
unlimited number of members and an unlimited number of collaborators simultaneously working on individual projects. The IP
ownership of all the projects in these organizations remains within the organization. There is no limit on the number of
backtesting, research, and live trading nodes these organizations can rent. They can produce 5MB of logs/backtest and 50MB
of logs/day. Members in these organizations can have up to eight active coding sessions in the organization. Each live algorithm
in a Trading Firm organization can send up to 240 Telegram, Email, or Webhook notifications per hour for free.
The owner of a Trading Firm organization can grant various permissions to the organizationʼs members, including designating a
member to manage the organization's billing. These organizations have access to custom lean builds, so they can use feature
branches or historical master branches to run their strategies. An example of this could be granting only a few members of your
team live trading deployment access.
In addition to the brokerages and data providers available to Team organizations, Trading Firm organizations can use Interactive
Brokers Financial Advisor accounts to manage sub-accounts for clients.
Institution Tier
The Institution tier is designed for established larger funds, large prop desks, hedge funds, banks, ETF vehicles, and
professional teams of quants. It is "unlocked", so you can run it on premise to serve your internal teams. If this sounds
interesting, reach out and we'd be happy to arrange a demonstration for your department.
The Institution tier builds on the features included in the Trading Firm tier. Organizations on the institutional tier have no limit on
the number of backtest logs that they can produce, and each member of the organization can have up to 16 active coding
sessions. These organizations can use Terminal Link to live trade Equities, Futures, and Options via the Bloomberg EMSX. They
can also request custom libraries and frameworks to use in the QuantConnect web IDE and receive instant messaging support
from a QuantConnect engineer. Each live algorithm in an Institution organization can send up to 3,600 Telegram, Email, or
Webhook notifications per hour for free.
Institutional clients can run our proprietary build of LEAN, LEAN Enterprise, on-premise. LEAN Enterprise holds speed
improvements we haven't pushed to the open-source.
Organizations > Resources
Organizations
Resources
Introduction
Organizations can subscribe to hardware resources to run backtests, launch research notebooks, and deploy live trading
algorithms to co-located servers. Organizations also have access to storage resources via the Object Store to store data
between backtests or live trading deployments. To promote efficiency, all of these resources within your organization are
shared among all of the members within the organization. A team of several quants can all share one backtest, research, and
live trading node.
Backtesting Nodes
Backtesting nodes enable you to run backtests. The more backtesting nodes your organization has, the more concurrent
backtests that you can run. Several models of backtesting nodes are available. Backtesting nodes that are more powerful can
run faster backtests and backtest nodes with more RAM can handle more memory-intensive operations like training machine
learning models, processing Options data, and managing large universes. The following table shows the specifications of the
backtesting node models:
Name Number of Cores
Processing Speed
(GHz)
RAM (GB) GPU
B-MICRO 2 3.3 8 0
B2-8 2 4.9 8 0
B4-12 4 4.9 12 0
B4-16-GPU 4 3 16 1/3
B8-16 8 4.9 16 0
Refer to the Pricing page to see the price of each backtesting node model. You get one free B-MICRO backtesting node in your
first organization. This node incurs a 20-second delay when you launch backtests, but the delay is removed and the node is
replaced when upgrade your organization to a paid tier and add a new backtesting node .
To view the status of all of your organization's nodes, see the Resources panel of the IDE. When you run a backtest, it uses the
best-performing resource by default, but you can select a specific resource to use .
The CPU nodes are available on a fair usage basis while the GPU nodes can be shared with a maximum of three members.
Depending on the server load, you may use all of the GPU's processing power. GPU nodes perform best on repetitive and
highly-parallel tasks like training machine learning models. It takes time to transfer the data to the GPU for computation, so if
your algorithm doesn't train machine learning models, the extra time it takes to transfer the data can make it appear that GPU
nodes run slower than CPU nodes.
You can't use backtesting nodes for optimizations .
Research Nodes
Research nodes enable you to spin up an interactive, command-line, Jupyter Research Environment . Several models of
research nodes are available. More powerful research nodes allow you to handle more data and run faster computations in your
notebooks. The following table shows the specifications of the research node models:
Name Number of Cores
Processing Speed
(GHz)
RAM (GB) GPU
R1-4 1 2.4 4 0
R2-8 2 2.4 8 0
R4-12 4 2.4 12 0
R4-16-GPU 4 3 16 1/3
R8-16 8 2.4 16 0
Refer to the Pricing page to see the price of each research node model. You get one free R1-4 research node in your first
organization, but the node is replaced when you subscribe to a paid research node in the organization.
To view the status of all of your organization's nodes, see the Resources panel of the IDE. When you launch the Research
Environment, it uses the best-performing resource by default, but you can select a specific resource to use .
The CPU nodes are available on a fair usage basis. The GPU nodes can be shared with a maximum of three members. Depending
on the server load, you may use all of the GPU's processing power.
Live Trading Nodes
Live trading nodes enable you to deploy live algorithms to our professionally-managed, co-located servers. You need a live
trading node for each algorithm that you deploy to our co-located servers. Several models of live trading nodes are available.
More powerful live trading nodes allow you to run algorithms with larger universes and give you more time for machine learning
training . Each security subscription requires about 5MB of RAM. The following table shows the specifications of the live trading
node models:
Name Number of Cores
Processing Speed
(GHz)
RAM (GB) GPU
L-MICRO 1 2.6 0.5 0
L1-1 1 2.6 1 0
L1-2 1 2.6 2 0
L2-4 2 2.6 4 0
L8-16-GPU 8 3.1 16 1/2
Refer to the Pricing page to see the price of each live trading node model.
To view the status of all of your organization's nodes, see the Resources panel of the IDE. When you deploy an algorithm, it uses
the best-performing resource by default, but you can select a specific resource to use .
The CPU nodes are available on a fair usage basis while the GPU nodes can be shared with a maximum of two members.
Depending on the server load, you may use all of the GPU's processing power. GPU nodes perform best on repetitive and
highly-parallel tasks like training machine learning models. It takes time to transfer the data to the GPU for computation, so if
your algorithm doesn't train machine learning models, the extra time it takes to transfer the data can make it appear that GPU
nodes run slower than CPU nodes.
Sharing Resources
Your organization's nodes are shared among all of the organization's members to reduce the amount of time that nodes idle. In
the Algorithm Lab, you can see which nodes are available within your organization.
Node Quotas
The following table shows the number of nodes each organization tier can have:
Tier Backtest Research Live Trading
Free 1 1 0
Quant Researcher 2 1 2
Team 10 10 10
Trading Firm Inf. Inf. Inf.
Institution Inf. Inf. Inf.
Training Quotas
Algorithms normally must return from the on_data method within 10 minutes, but the train method lets you increase this
amount of time. Training resources are allocated with a leaky bucket algorithm where you can use a maximum of n-minutes in a
single training session and the number of minutes available refills over time. This gives you a reservoir of training time when you
need it and recharges the reservoir to prepare for the next training session. The reservoir only starts draining after you exceed
the standard 10 minutes of training time.
The following animation demonstrates the leaky bucket algorithm. The tap continuously adds water to the bucket. When the
bucket is full, water spills over the rim of the bucket. The water represents your training resources. When your algorithm
exceeds the 10 minutes of training time, holes open at the bottom of the bucket and water begins to drain out. When your
algorithm stops training, the holes close and the bucket fills up with water.
The following table shows the amount of extra time that each backtesting and live trading node can spend training machine
learning models:
Model Capacity (min) Refill Rate (min/day)
B-MICRO 20 1
B2-8 30 5
B4-12 60 10
B8-16 90 15
L-MICRO 30 5
L1-1 60 10
L1-2 90 15
L1-4 120 20
The refill rate in the table above is based on the real-world clock time, not the backtest clock time. In backtests, the train
method is synchronous, so it will block your algorithm from executing while the model is trained. In live trading, the method runs
asynchronously, so ensure your model is ready to use before you continue executing the algorithm. Training occurs on a
separate thread, so use a semaphore to track the model state.
Log Quotas
Per our Terms and Conditions , you may not use the logs to export dataset information. The following table shows the amount of
logs each organization tier can produce:
Tier Logs Per Backtest Logs Per Day
Free 10KB 3MB
Quant Researcher 100KB 3MB
Team 1MB 10MB
Trading Firm 5MB 50MB
Institution Inf. Inf.
To check the log storage space you have remaining, log in to the Algorithm Lab and then, in the left navigation bar, click
Organization > Resources .
If you delete a backtest or project that produced logs, your quotas aren't restored. Additionally, daily log quotas aren't fully
restored at midnight. They are restored according to a 24-hour rolling window.
The log files of each live trading project can store up to 1,000,000 lines for up to two years. If you log more than 1,000,000 lines
or some lines become older than two years, we remove the oldest lines in the files so your project stays within the quota.
To avoid reaching the limits, we recommend logging sparsely, focusing on the change events instead of logging every time loop.
You can use the debugger to inspect objects during runtime. If you use the debugger, you should rarely reach the log limits.
Coding Session Quotas
If you have a project open, it uses a coding session. Paid organizations can have multiple active coding sessions, but free users
can only have one coding session open at a time. The following table shows how many active coding sessions you can have on
each organization tier:
Tier Initial Coding Session Quota
Quant Researcher 2
Team 4
Trading Firm 8
Institution 16
If the organization you're in has more live trading nodes than your initial coding session quota, then your coding session quota
increases to the number of live trading nodes you have in the organization so you can view all your live strategies.
The quota for free organizations is a global quota, so you can have one active coding session across all of your free
organizations. The quotas for paid organizations are at the organization level. Therefore, if you are in two Quant Researcher
organizations, you can have two active coding sessions in one of those organizations and another two active sessions in the
other organization. These paid tier quotas are for each account, not for the organization as a whole. For instance, a Trading Firm
organization can have more than eight members and all of the members can simultaneously work on projects within the
organization.
File Size Quotas
The maximum file size you can have in a project depends on your organization's tier. The following table shows the quota of
each tier:
Tier Max File Size (KB)
Free 32
Quant Researcher 64
Team 128
Trading Firm 256
Institution 256
Live Trading Notification Quotas
The number of email, Telegram, or webhook notifications you can send in each live algorithm for free depends on the tier of
your organization. The following table shows the hourly quotas:
Tier Number of Notifications Per Hour
Free N/A
Quant Researcher 20
Team 60
Trading Firm 240
Institution 3,600
If you exceed the hourly quota, each additional email, Telegram, or webhook notification costs 1 QuantConnect Credit (QCC).
Each SMS notification you send to a US or Canadian phone number costs 1 QCC. Each SMS notification you send to an
international phone number costs 10 QCC.
View All Nodes
The Resources page displays your backtesting, research, and live trading node clusters. To view the page, log in to the
Algorithm Lab and then, in the left navigation bar, click Organization > Resources .
To toggle the format of the page, click the buttons in the top-right. If the page is in table view, each cluster section includes a
table with the following columns:
Column Description
Name Name of the node
Machine Type The node model and specifications
In Use By The name of the member using the node
Host The live trading server name
Assets
The recommended maximum number of assets to avoid
RAM errors.
Actions A list of possible actions
Add Nodes
You need billing permissions in the organization to add nodes.
Follow these steps to add nodes to your organization:
1. Open the Resources page.
2. Click Add nodeType Node for the type of node you want to add.
3. Select the node specifications.
4. Click Add Node .
The Resources page displays the new node.
Remove Nodes
You need billing permissions in the organization to remove nodes. If you remove nodes during your billing period, your
organization will receive a pro-rated credit on your account, which is applied to future invoices.
Follow these steps to remove nodes from your organization:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Organization > Home .
3. On the organization homepage, click Edit Plan .
4. Click the Customize Plan > Build Your Own Pack > Compute Nodes tab.
5. Click the minus sign next to the node model you want to remove.
6. Click Proceed to Checkout .
Rename Nodes
We assign a default name to hardware nodes that includes the model name and an arbitrary string of characters, but you can
follow these steps to rename the nodes in your organization:
1. Open the Resources page.
2. Click Set Name on the node that you want to rename.
3. Enter the new node name and then click Save Changes .
The Resources page displays the new node name.
Stop Nodes
You need stop node permissions in the organization to stop nodes other members are using. If you stop a node, it terminates
the running backtest, research, or live trading sessions. When you stop a live trading node, the portfolio holdings don't change
but the algorithm stops executing.
Follow these steps to stop nodes that are running in your organization:
1. Open the Resources page.
2. Click the icon with the three horizontal lines icon in the top-right corner to format the page into table view.
3. Click Stop in the row with the node that you want to stop.
Organizations > Object Store
Organizations
Object Store
Introduction
The Object Store is an organization-specific key-value storage location to save and retrieve data in QuantConnect's cache.
Similar to a dictionary or hash table, a key-value store is a storage system that saves and retrieves objects by using keys. A key
is a unique string that is associated with a single record in the key-value store and a value is an object being stored. Some
common use cases of the Object Store include the following:
Transporting data between the backtesting environment and the research environment.
Training machine learning models in the research environment before deploying them to live trading.
The Object Store is shared across the entire organization. Using the same key, you can access data across all projects in an
organization.
View Storage
The Object Store page shows all the data your organization has in the Object Store. To view the page, log in to the Algorithm Lab
and then, in the left navigation bar, click Organization > Object Store .
To view the metadata of a file (including it's path, size, and a content preview), click one of the files in the table.
Upload Files
Follow these steps to upload files to the Object Store:
1. Open the Object Store page.
2. Navigate to the directory in the Object Store where you want to upload files.
3. Click Upload .
4. Drag and drop the files you want to upload.
Alternatively, you can add data to the Object Store in an algorithm or notebook .
Download Files
Permissioned Institutional clients can build derivative data such as machine learning models and download it from the Object
Store. Contact us to unlock this feature for your account.
Follow these steps to download files and directories from the Object Store:
1. Open the Object Store page.
2. Navigate to the directory in the Object Store where you want to download files and directories.
3. Select the file(s) and directory(ies) to download
4. Click Download .
5. Wait while QuantConnect processes the request.
6. Click the Download link that appears.
Storage Sizes
All organizations get 50 MB of free storage in the Object Store. Paid organizations can subscribe to more storage space. The
following table shows the cost of the supported storage sizes:
Storage Size (GB) Storage Files (-) Monthly Cost ($)
0.05 1,000 0
2 20,000 10
5 50,000 20
10 100,000 50
50 500,000 100
Delete Storage
Follow these steps to delete storage from the Object Store:
1. Open the Object Store page.
2. Navigate to the directory in the Object Store where you want to delete files.
3. Click the check box next to the files you want to delete.
4. Click Actions and then click Delete from the drop-down menu.
5. Click OK .
Alternatively, you can delete storage in an algorithm or notebook .
Edit Storage Plan
You need storage billing permissions and a paid organization to edit the size of the organization's Object Store.
Follow these steps to edit the amount of storage available in your organization's Object Store:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Organization > Resources .
3. On the Resources page, scroll down to the Storage Resources and then click Add Object Store Capacity .
4. On the Pricing page, select a storage plan.
5. Click Proceed to Checkout .
Research to Live Considerations
When you deploy a live algorithm, you can access the data within minutes of modifying the Object Store. Ensure your algorithm
is able to handle a changing dataset.
The live environment's access to the Object Store is much slower than in research and backtesting. Limit the individual objects
to less than 50 MB to prevent live trading access issues.
Usage by Project
The Resources page shows the total storage used in your organization and the storage used by individual projects so that you
can easily manage your storage space. To view the page, log in to the Algorithm Lab and then, in the left navigation bar, click
Organization > Resources .
Organizations > Support
Organizations
Support
Introduction
The community is a great resource for support for developing algorithms. However, for more personalized assistance and
privacy, you can submit support tickets to request assistance from a QuantConnect engineer. Premium support allows you to
share your algorithms with IP protection and enables the Support Team to address issues with live algorithms. In order to submit
a support ticket, you must have a support seat in your organization.
There are three tiers of support seats and each tier provides different services. You can file support tickets to get private
assistance with issues, but support tickets should not replace your efforts of performing your own research and reading through
the documentation. If you need further assistance than what our Support Team offers, consider hiring an Integration Partner .
Features
The services our Support Team provides depend on the tier of your support seat.
Bug Reports
The Lean trading engine is under constant development, so you may occasionally experience bugs when using it. If you think
you have found a bug, share a simple backtest with us that reproduces the issue. You can contact us either through email ,
Discord , or the forum . We will review your submission. If we confirm you've found a bug, we will create a GitHub Issue to have
it resolved. Subscribe to the GitHub Issue to track our progress in fixing the bug.
Feature Community Bronze Silver Gold
Bug Reports
AI Support
Mia is an AI assistant we trained on hundreds of algorithms and thousands of documentation pages to provide contextual
assistance for most issues you may encounter when developing a strategy.
Feature Community Bronze Silver Gold
AI Support
Email Support
Our support ticket system enables you to privately email with our Support Team. We address support tickets in a first-in, firstout order, but we give priority to tickets opened by members with higher support seats. The following table shows our response
time for each of the support tiers:
Tier Response Time (hours)
Gold 24
Silver 48
Bronze 72 or Best Effort
Feature Community Bronze Silver Gold
Email Support -
IP Protection
If you attach a backtest or live trading deployment when you open a support ticket , the intellectual property of your project is
protected. We have a restricted subset of the Support Team who can access private support tickets. Using paid support plans
ensures only a limited subset of the QuantConnect team can access the algorithms you attach to support tickets. These team
members are carefully selected, have been at QuantConnect for at least 2 years, and have passed a background check. In
contrast, if you share a backtest or live trading deployment to the forum for assistance, your project becomes part of the public
domain.
Feature Community Bronze Silver Gold
IP Protection -
Live Trading Debugging
Paid support plans have access to our live deployment debugging service. If you experience an issue with a live trading
deployment, open a support ticket. We can assist with uncovering the issue, fixing the issue, and getting you ready to redeploy
the algorithm. We can't assist with live trading issues in the community forum or Discord.
Feature Community Bronze Silver Gold
Live Trading Debugging -
Algorithm Design Suggestions
If you have a silver or gold support seat, we can offer suggestions on the design of your algorithms. Our Support Team
members are experts on the inner workings of Lean, so we can guide you on improving the efficiency of your algorithms by
following our common design patterns. We can usually reduce the size of your project's code files and increase the speed of
your project backtesting.
Feature Community Bronze Silver Gold
Algorithm Design Suggestions - -
Private Chatroom
When you require instant access to our Support team, we can open a private chatroom in Discord. In our private chatroom, you
can ask our Support Team questions at any time and we focus on responding as quickly as we can. Request a private chatroom
to avoid waiting for email responses on your support tickets.
Feature Community Bronze Silver Gold
Private Chatroom - - -
Phone Call Consultations
We offer phone support to members with gold support seats. You can take advantage of our phone support for up to 1 hour per
month. During phone calls, feel free to ask about anything related to QuantConnect, Lean, or quant trading. We recommend
planning your questions before you call in order to best utilize the time available.
Feature Community Bronze Silver Gold
Phone Call Consultations - - -
Local Development
If you have a gold support seat, you can access technical support for the Local Platform , LEAN CLI , and the QuantConnect
REST API , which includes installation and subjects that don't apply to the Cloud Platform.
Feature Community Bronze Silver Gold
Private Chatroom - - -
Summary
The following table shows the features available in each tier:
Feature Community Bronze Silver Gold
Bug Reports
AI Support
Email Support -
IP Protection -
Live Trading Debugging -
Algorithm Design Suggestions - -
Private Chatroom - - -
Phone Call Consultations - - -
Local Development - - -
View All Seats
The Team Management page displays the support seat assignments within your organization. To view the page, log in to the
Algorithm lab, and then in the left navigation bar, click Organization > Members .
Add Seats
Follow these steps to assign support seats to members in your organization:
1. Open the Team Management page.
2. Scroll down to the Organization Support section and then click Add Seat .
3. On the Support page, click Select under the tier of seat you want to assign.
4. In the Assign seatTier Seat section, click the team member field and then select the team member you want to assign the
seat to from the drop-down menu.
5. Click Add Seat +$ price .
View All Tickets
The Support History page displays all of your organization's support tickets. To view the page, log in to the Algorithm Lab and
then, in the left navigation bar, click Support > Organization Tickets .
To see all closed tickets, click Closed . To view the conversation history with our Support Team regarding a ticket, click a ticket.
Open New Tickets
Follow these steps to open support tickets:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Support .
3. On the Support page, enter a subject and message.
4. If you want to attach a live deployment or backtest to the support ticket, follow these steps:
1. Click Add Project , select a project, and then click OK .
2. (Optional) Click Add Live Deployment , select a live deployment, and then click OK .
3. (Optional) Click Add Backtest , select a backtest, and then click OK .
5. Click the By submitting this request, I give QuantConnect Support Staff permission to view my project check box.
6. Click Send .
Mia, our AI assistant, provides an initial response to support tickets. We trained it on hundreds of algorithms and thousands of
documentation pages to provide contextual assistance for most issues you may encounter when developing a strategy. To
escalate the ticket to a human on the Support Team, click No, I'd like a human response .
Comment on Tickets
Follow these steps to comment on support tickets:
1. Open the Support History page.
2. If the ticket you want to comment on is closed, click Closed .
3. Click the ticket on which you want to comment.
4. Enter your comment and then click Send .
Close Tickets
Follow these steps to close support tickets:
1. Open the Support History page.
2. Click the ticket you want to close.
3. (Optional) Enter a reason for closing the ticket.
4. Click Send and Close Ticket .
Open Closed Tickets
Follow these steps to open closed support tickets:
1. Open the Support History page.
2. Click Closed .
3. Click the ticket that you want to open.
4. (Optional) Enter a reason for opening the ticket.
5. Click Open Ticket and Send .
Hours of Operation
The human Support Team operates Monday to Friday. Mia, our AI support agent, operates 24/7.
AI Support
Mia is an AI assistant we trained on hundreds of algorithms and thousands of documentation pages to provide contextual
assistance for most issues you may encounter when developing a strategy. Mia automatically provides an inital response to the
support tickets you open. You can also interact with Mia through the community forum , Discord , or the Algorithm Lab Console .
Ticket Quotas
The following table shows the number of tickets each seat tier can open during a 1-month rolling period:
Tier Human Ticket Quota AI Ticket Quota
Gold 16 250
Silver 8 125
Bronze 4 25
Community - 3
The quotas don't restore on the first day of the month or the billing date. They restore according to a 1-month rolling window.
Organizations > Members
Organizations
Members
Introduction
Organizations consist of members and members can be part of multiple organizations. The number of members your
organization can have depends on the tier of your organization. All members in your organization can access the resources
within the organization.
Members can collaborate with other team members by adding them to projects they create. Otherwise, members don't have
access to projects in your organization.
If you are the manager of a Trading Firm or Institution organization, you can grant additional permissions to your team members.
View All Seats
You need a seat for each member in your organization.
To see the number of team seats you have in your organization, open the Billing page and then scroll down to the Products
section.
Add Seats
You need billing permissions within an organization to add team seats.
Follow these steps to add team seats to your organization:
1. Open the organization homepage .
2. Click Edit Plan .
3. Click the Customize Plan > Build Your Own Pack > Organization Seats tab.
4. In the organizationTier Seats section, click the plus icon.
5. Click Proceed to Checkout .
View All Members
The Team Management page displays all of the members in your organization. To view the page, log in to the Algorithm Lab and
then, in the left navigation bar, click Organization > Members .
To toggle the format of the page, click the buttons in the top-right.
Membership Quotas
The number of members your organization can have depends on the organization's tier . In general, higher organization tiers
can have more members within the organization to share resources. This design enables you to upgrade your organization as
your trading business grows over time. The following table shows the number of members each organization tier can have:
Tier Minimum Members Maximum Members
Free 1 1
Quant Researcher 1 1
Team 2 10
Trading Firm 2 Unlimited
Institution 2 Unlimited
Add Members
You need vacant team seats and team add permissions within an organization to add team members.
Follow these steps to add team members to your organization:
1. Open the Team Management page.
2. Click Add Member .
3. Enter the email address of the new team member and then click Add Member .
The email you enter should be the email the new member uses to log in to their QuantConnect account.
4. If you have team edit permissions, select the permissions for the new member and then click Save Changes . Otherwise,
click Cancel .
The Team Management page displays and the new member is included.
On the Add Member dialog you can generate a unique link to invite team members to your organization. Anyone with this link
can join your organization.
The new member must switch organization to access its resources and collaborate.
Add Support
Follow these steps to assign support seats to members in your organization:
1. Open the Team Management page.
2. Scroll down to the Organization Support section and then click Add Seat .
3. On the Support page, click Select under the tier of seat you want to assign.
4. In the Assign seatTier Seat section, click the team member field and then select the team member you want to assign the
seat to from the drop-down menu.
5. Click Add Seat +$ price .
Remove Members
You need team removal permission to remove members from your organization. Before removing members, stop all nodes they
are using. Removing members is not reversible and the members will lose access to the organization's projects. On the Team
tier, projects that the members created in the organization will remain with the members. On the Trading Firm and Institution
tiers, projects that the members created in the organization will be transferred to the organization manager.
Follow these steps to remove members from your organization:
1. Open the Team Management page.
2. If the Team Management page is in tile view, follow these steps:
1. Click the three dots icon in the top-right corner of the member you want to remove and then click Remove .
2. Click Remove .
3. If the Team Management page is in table view, follow these steps:
1. Click Remove next to the member you want to remove from the organization.
2. Click Remove .
Permissions
Each member of your organization has access to the resources within the organization. If you are the manager of a Trading Firm
or Institution organization, you can grant additional permissions to your team members. There are several categories of
permissions.
Billing Permissions
The following table shows the supported billing permissions:
Permission Description
Update Permission to change the organization's subscriptions and billing details.
Stop Node Permissions
The following table shows the supported node permissions:
Permission Description
Backtest Permission to stop backtesting nodes .
Research Permission to stop research nodes .
Live Permission to stop live trading nodes .
To stop active nodes, see Stop Nodes .
Team Permissions
The following table shows the supported team permissions:
Permission Description
Add Permission to add new members .
Remove Permission to remove existing members .
Edit Permission to change the permissions of other members .
Storage Permissions
The following table shows the supported team permissions:
Permission Description
Create Permission to write to the Object Store .
Delete Permission to delete data in the Object Store .
Billing Permission to subscribe to more space in the Object Store .
Edit Permissions
You need team edit permission to edit the permissions of team members.
Follow these steps to edit team permissions:
1. Open the Team Management page.
2. If the Team Management page is in tile view, follow these steps:
1. On the Team Management page, click the three dots in the top-right corner of the member you want to edit and then
click Edit Permissions .
2. Select and deselect permissions as desired and then click Save Changes .
3. If the Team Management page is in table view, follow these steps:
1. On the Team Management page, click Edit Permissions next to the member whose permissions you want to edit.
2. Select and deselect permissions as desired and then click Save Changes .
Organizations > Administration
Organizations
Administration
Introduction
You can view and manage your organizations from the Algorithm Lab. The algorithms you store in the Algorithm Lab are secure
and you maintain their intellectual property.
Intellectual Property
All individuals on QuantConnect own their intellectual property (IP) on our platform. Your code is private and only accessible by
people you share the project with and with support-engineers when you submit a support ticket. At no point does QuantConnect
ever claim ownership of user IP. The only case where algorithm code becomes public domain is when they are shared to the
forum. In this case, algorithms need to be made public domain to allow the sharing of the algorithm code.
It is common when companies hire engineers to write software, they require their employees to sign an agreement that gives the
company IP ownership of any code written for work. They need this because they're paying you to write software, and the
company needs to sell that software to turn a profit. Similarly, the Organizations feature allows you to control who holds IP
ownership over a project. Each type of organization has its own mechanisms for handling project IP ownership.
Individual Organizations
The Free and the Quant Researcher tiers only allow single-member organizations. This means you can't collaborate with anyone
else inside the QuantConnect platform. Simply put, you own the IP for any projects you work on since you are the sole
collaborator.
Team Organizations
For organizations that allow multiple users to collaborate on projects, the user who created the project owns it; this can be you
or one of your teammates. If you add a teammate/collaborate, they can clone it, but the original project belongs to the person
who first created it.
Trading Firm & Institution Organizations
For Trading Firm and Institution organizations, which are generally used by companies and funds, the firm owns all employee
projects. This is made to suit firms that wish to hire consultants and need to ensure the code remains with the company when
the consultant work is finished. You have to explicitly create a project in an organization for it to be created on the
organization's account.
Corporate Branding
You can customize your organizationʼs image, name, and description in the Algorithm Lab to match your branding. If you have a
Trading Firm or Institution organization, you can integrate the Algorithm Lab into your website so that your company logo is in
the navigation bar and the color matches your website's theme.
Migrating Projects
If you are the administrator of an organization, you can migrate a project out of the organization to another organization in which
youʼre a member. When your project is migrated, the project files are copied but the content stored in the Object Store is not
retained.
View the Organization Homepage
The organization homepage displays a summary of your organization. To view the page, log in to the Algorithm Lab and then, in
the left navigation bar, click Organization > Home .
The organization homepage displays your organization's brand and statistics at the top of the page. The following table
describes the remaining sections of the page:
Section Description
Actions Add nodes
Resources View and manage nodes
Billing View and manage bills
Team View and manage team members
Support History View support tickets
Encryption Keys View and manage encryption keys
Backtesting Out of Sample Period View and change the length of the out-of-sample hold out
period
Plan View and change the organization tier
Credit Balance View and purchase QuantConnect Credit
Edit the Organization Branding
You can edit your organization's image and name.
Image
Follow these steps to change your organization image:
1. Open the organization homepage .
2. Click the organization image.
3. Click Choose file , select a file from your device, and then click Open .
Your organization image must be in gif , jpg , or png format and less than 1MB in size.
4. Click Save .
"Photo Uploaded" displays.
Name
Follow these steps to change your organization name:
1. Open the organization homepage .
2. Hover over the organization name and then click the pencil icon that appears.
3. Enter the new organization name and then click Save Changes .
"Organization Name Updated Successfully" displays.
View All Organizations
To view all of the organizations for which you're a member, log in to the Algorithm lab and then, in the top navigation bar, click
Connected as: organizationName .
Add Organizations
Follow these steps to add new organizations to your profile:
1. Log in to the Algorithm Lab.
2. In the top navigation bar, click Connected as: organizationName .
3. In the Switch Organization panel, click Create Organization .
4. Enter the organization name and then click Add .
The organization name must be unique. "Created Successfully" displays.
Switch Organizations
Follow these steps to switch organizations:
1. Log in to the Algorithm Lab.
2. In the top navigation bar, click Connected as: organizationName .
3. In the Switch Organization panel, click the name of the organization for which you want to connect.
The top navigation bar displays the new organization name.
Set a Preferred Organization
Follow these steps to set your preferred organization:
1. Log in to the Algorithm Lab.
2. In the top navigation bar, click Connected as: organizationName .
3. In the Switch Organization panel, select the radio button under the Preferred column that corresponds to the organization
that you want to set as the preferred organization.
"Preferred organization selected" displays. Refresh the page to connect as your preferred organization.
Get Organization Id
To get the organization Id, open Organization > Home and check the URL. For example, the organization Id of
https://www.quantconnect.com/organization/5cad178b20a1d52567b534553413b691 is 5cad178b20a1d52567b534553413b691.
Out of Sample Period
To reduce the chance of overfitting, organization managers can enforce all backtests must end a certain number of months
before the current date. For example, if you set a one year out-of-sample period, the researchers on your team will not be able
to use the most recent year of data in their backtests. A out-of-sample period is helpful because it leaves you a period to test
your model after your done the development stage. Follow these steps to change the backtest out-of-sample period:
1. Open the organization homepage .
2. Scroll down to the Backtesting Out of Sample Period section.
3. Adjust the out-of-sample period duration or click on "No Holdout Period".
Organizations > Billing
Organizations
Billing
Introduction
The owner of an organization is responsible for the billing, but the responsibility can be delegated in Trading Firm or Institution
organizations. Your organization's billing information is never saved by QuantConnect. Itʼs passed to the Stripe billing system. If
you cancel your subscription, your live trading nodes stop running. So, for user safety, your subscriptions automatically renew
each month.
View Billing Information
The Billing page displays your organization's billing details. To view the page, log into the Algorithm Lab and then, in the left
navigation bar, click Organization > Billing .
The Billing page displays the billing cost, date, and frequency at the top of the page. The following table describes the
remaining sections of the page:
Section Description
Details
The billing name and address associated with the credit
card.
Credit Card The credit card used to pay the bill.
Products A breakdown of the organization's subscriptions.
Invoices All the organization's invoices.
Organization Credit (QCC) All QCC purchases and expenses.
Edit Billing Details
Follow these steps to edit your organization's billing details:
1. Open the Billing page.
2. In the Details section, click Edit .
3. Enter the new billing details and then click Save .
The Details section displays the new billing details.
Edit the Credit Card
You can add, replace, and remove the organization's credit card.
Add
Follow these steps to add a credit card:
1. Open the Billing page.
2. Click Add Card .
3. Enter the credit card details and then click Save .
The Credit Card section displays the last 4 digits of your credit card.
Replace
Follow these steps to change the credit card:
1. Open the Billing page.
2. In the Credit Card section, click Edit .
3. Enter the new credit card details and then click Save .
The Credit Card section displays the last 4 digits of your new credit card.
Remove
Follow these steps to remove the credit card:
1. Open the Billing page.
2. Click Remove .
3. Click OK .
The Credit Card section displays "No entries found".
Download Invoices and Receipts
Follow these steps to download invoices and receipts:
1. Open the Billing page.
2. Scroll down to the Invoices section and then click Download as PDF next to the invoice or receipt that you want to
download.
3. Click Download invoice or Download receipt .
A prompt to download the file to your local machine displays.
Change Organization Tiers
You can change your organization to any of the paid tiers or the Free tier.
Paid Tiers
Follow these steps to change to a paid organization tier:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Organization > Home .
3. On the organization homepage, click Edit Plan .
4. Click the Choose a Plan tab.
5. Click CHOOSE TIER under the organization tier you want.
6. Select a tier pack.
The following table describes the type of packs we have available:
Pack Type Description
Suggested Packs
Packs with pre-selected team seats, support seats, and
nodes.
Build Your Own Pack
Packs with custom selections for team seats, support
seats, add-ons, and market subscriptions.
7. Select monthly or annual billing.
8. (Optional) Click + Add Coupon and then enter your coupon code.
9. If your organization doesn't have a credit card added, click Proceed to Checkout and then enter your credit card details.
10. Click Update Subscriptions or Subscribe Now .
Free Tier
Follow these steps to downgrade to the Free tier:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Organization > Home .
3. On the organization homepage, click Downgrade to Free .
4. Click Cancel My Plan .
5. Click the I understand that my subscription will be terminated immediately check box and then click Cancel My
Subscription .
Pause and Resume Subscriptions
If you're not going to utilize your QuantConnect subscriptions for a month or two, pause your subscriptions to avoid
unnecessary charges. Pausing suspends access to your account, but all your cloud configuration, project files, and set up will
be kept as is. When you resume your subscriptions, your account will be the same as you left it.
Pause Subscriptions
Follow these steps to pause your subscriptions:
1. Open the Billing page.
2. In the Details section, click Cancel .
3. On the Downgrade page, click Pause for 1 month or Pause for 2 months .
4. Click Pause My Subscription .
Resume Subscriptions
To resume your paused subscriptions, open the Billing page and then click Resume . If you prematurely resume your
subscriptions, the credit you receive is pro-rated based on the duration of the pause period.
Payment Failures
If your payment fails, Stripe automatically tries again after 6-12 hours.
Organizations > Credit
Organizations
Credit
Introduction
We created QuantConnect Credit (QCC) to enable micropayments on QuantConnect. You can use QCC to optimize parameters,
download datasets, gift to members in the forum, and apply to your organization's monthly invoice. You can purchase QCC in
the Algorithm Lab at a rate of 1 QCC = $0.01 USD. Since QCC is owned by organizations instead of members, all of the members
within your organization have the ability to spend the QCC balance.
Optimizing Parameters
You need QCC in your organization to unlock parameter optimization . Parameter optimization jobs use optimization nodes,
which are rented on a time basis. Therefore, the longer it takes to run all of the backtests in your optimization job, the more QCC
it costs. Before you run optimizations, we estimate on how much QCC it will cost to run the job, but the final cost can differ from
our estimates because we can't know exactly how long all of the backtests will take to run ahead of time. For instructions on
optimizing parameters, see Launch Optimization Jobs .
Downloading Datasets
You can spend some QCC to download datasets from the Dataset Market to your local machine. The cost of downloading
depends on the dataset and it's calculated on a per-file or per-day basis. For instructions on downloading datasets, see
Licensing .
Giving to Others
To show your appreciation for contributions in the forum, give some QuantConnect Credit (QCC) rewards . The following table
shows the available QCC rewards:
Award Description Cost (QCC)
Silver Award
A simple token of recognition from one
quant to another. Keep up the great
work.
80
Gold Award
This is some great work! Gold star! 600
Platinum Award
Highly resistant to oxidation, this
award is for those contributions which
will stand the test of time. Strong,
classic, and useful for most high
technology products.
1,200
Medal for Excellence
The QuantConnect Medal for
Excellence is awarded by a member of
the QuantConnect staff for
exceptional contributions to the
QuantConnect community.
3,000
Plutonium Award
Nuclear hot! This post is incredible
and deserves recognition as such.
Show the author your appreciation for
their work.
2,000
Docs Shakespeare
You've left a mark by a contribution to
the documentation for the community.
Your edits and examples will be
followed for generations to come.
500
Nobel Laureate
Bestowed in recognition of
quantitative advances.
80
Spaghetti Code Award
Following the intricate flows of code
noodle, this code compiles and runs.
300
Jedi Quant
Quant promise you show. Your code
channel the force into. Hmmmmmm.
200
Live Trader
That looks like a wild ride. 50
Today I Learned
Thank you for upgrading my brain. 50
Machine Unlearning
I plan on letting the computers do all
the work for me.
150
Totally Overfit
I see parameters everywhere. 80
Mind Blown Award
Something incredibly amazing, mindboggling, and you're shocked
senseless.
80
Research Rembrandt
A Jupyter notebook work of art,
pulling together all the right hues and
plots to be a true masterpiece.
100
Cayman Island Award
Let's get a boat and start a fund
together. Did I mention the boat?
800
Printing Money Award
Awarded to profitable algorithms. 100
Stronger Together Award
We're stronger together. Let's make
this happen!
120
Appreciate the Support Award
Quant trading is a hard road, we could
all use a hand.
250
Master Craftsman
You're a master craftsman, taking raw
materials and molding them into works
of art for the good of the world.
600
Purchase QCC
Follow these steps to purchase QuantConnect Credit (QCC):
1. Open the Billing page.
2. Scroll down to the Organization Credit (QCC) section and then click Purchase Credit .
3. Select a credit pack and then click Continue .
4. If your preferred organization has a credit card and you want to charge that card, click Purchase .
5. If your preferred organization doesn't have a credit card, enter the credit card details and then click Purchase .
Only the preferred organization can be charged when purchasing QCC. If you want to charge a different organization you
are a member of, set it as your preferred organization .
Apply QCC to Invoices
You can apply QCC to your organization's invoices to pay for your subscriptions, but you must enable invoice payments before
your invoice is generated to pay it with QCC. To enable invoice payment with QCC, open the Billing page, scroll down to the
Organization Credit (QCC) section, and then select the Only apply organization QCC to invoices check box.
Organizations > Training
Organizations
Training
Introduction
Onboard new team members to your organization through the content in the Learning Center . The Learning Center enables you
to systematically track and monitor the progress of your team members on the courses you purchase or create. If you purchase
or create a course, you can access it from the Organization > Resources page in the Algorithm Lab.
Paid Courses
Currently, there are only free tutorials in the Learning Center, but we expect to expand the paid course offerings through 2023.
We charge for paid courses based on the number of members in your organization. For example, if there are 5 team members in
your organization, multiply the listed price of a course by 5. When you purchase a course, the team members in your
organization have lifetime access to the course.
Private Courses
Private courses are not for sale in the Learning Center. You can upload private courses to your Organization > Resources page,
but they will not be available for other organizations to purchase. Create private courses to help onboard your team members
while using the familiar Learning Center environment. Since the Learning Center has built-in tools to help you monitor your team
members, you can track the progress of your team members as they work on your internal private courses.
This feature is designed for Institutional clients with large teams of quants on QuantConnect
Learning Center
Learning Center
The Learning Center is a coding environment to learn quantitative trading using LEAN. The Learning Center features a collection
of courses from educators from the QuantConnect team and the community. The goal of the Learning Center is to give you an
understanding of robust algorithm design and the tools you need to implement your own trading strategies. As you work through
the courses, youʼll manage a portfolio, use indicators in technical trading strategies, trade on universes of assets, automate
trades based on market behavior, and understand how data moves through your algorithm. After you've completed a course,
you can keep the code to perform further research and deploy to live trading. Start learning today because you need to
complete 30% of the Bootcamp lessons to post in the community forum.
Training
Get team members up to speed
Educators
Add courses to the Learning Center
Course Structure
Structured into digestible portions
See Also
Available Courses
Learn Programming
Algorithm Engine
Learning Center > Training
Learning Center
Training
Introduction
The Learning Center is a coding environment to learn quantitative trading using LEAN. The Learning Center features a collection
of courses from educators from the QuantConnect team and the community. The goal of the Learning Center is to give you an
understanding of robust algorithm design and the tools you need to implement your own trading strategies. As you work through
the courses, youʼll manage a portfolio, use indicators in technical trading strategies, trade on universes of assets, automate
trades based on market behavior, and understand how data moves through your algorithm. After you've completed a course,
you can keep the code to perform further research and deploy to live trading. Start learning today because you need to
complete 30% of the Bootcamp lessons to post in the community forum.
Onboard new team members to your organization through the content in the Learning Center. The Learning Center enables you
to systematically track and monitor the progress of your team members on the courses that you purchase or create. If you
purchase or create a course, you can access it from the Organization > Resources page in the Algorithm Lab.
Articles
QuantConnect maintains collections of related tutorials we call a Learning Series . We have tutorial series covering the topics
below - each with a set of articles or tutorials:
Investment Strategy Library
The Strategy Library is a collection of tutorials written by the QuantConnect team and community members. Review these
tutorials to learn about trading strategies found in the academic literature and how to implement them with QuantConnect/LEAN.
Introduction to Financial Python
Introduces basic Python functionality in the context of quantitative finance.
Introduction to Options
Introduces Options to those who are Option novices and have basic knowledge of applied mathematics, statistics, and financial
markets.
Applied Options
Simple Options trading algorithms on QuantConnect for those who already have basic knowledge of Options markets.
Paid Courses
Currently, there are only free tutorials in the Learning Center, but we expect to expand the paid course offerings through 2023.
We charge for paid courses based on the number of members in your organization. For example, if there are 5 team members in
your organization, multiply the listed price of a course by 5. When you purchase a course, the team members in your
organization have lifetime access to the course.
Private Courses
Private courses are not for sale in the Learning Center. You can upload private courses to your Organization > Resources page,
but they will not be available for other organizations to purchase. Create private courses to help onboard your team members
while using the familiar Learning Center environment. Since the Learning Center has built-in tools to help you monitor your team
members, you can track the progress of your team members as they work on your internal private courses.
This feature is designed for Institutional clients with large teams of quants on QuantConnect
View All Courses
The Available Courses page displays all of the courses in the Learning Center. To view the page, open the Algorithm Lab and
then, in the left navigation bar, click Learning Center > All Courses .
Each course displays the following information:
Name
Description
Author
Price
Review summary
The number of students
Click a course to learn more about it, including the following:
Instructor biography
Requirements
Syllabus
Reviews
Enroll in Courses
Follow these steps to enroll in Learning Center courses:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Learning Center > All Courses .
3. On the Available Courses page, click the course in which you want to enroll.
4. Click Enroll .
The Learning Center environment displays.
Navigate the Course IDE
The course IDE automatically displays when you enroll in a course.
Follow these steps to navigate the course IDE:
1. Read the instructions in the left panel.
2. Update the main.py file with your answer.
3. (Optional) Scroll down to the bottom of the instruction panel and click Show Hint to show a hint.
A hint displays at the bottom of the instruction panel.
4. (Optional) Scroll down to the bottom of the instruction panel and click Solution to show the solution file.
A solution.py file displays.
5. (Optional) Click Reset to reset the main.py file.
6. Click Submit to check your answer.
The Chart panel displays your backtest results.
7. If an error message displays, restart from step 2.
8. Click Continue .
View Course Progress
Follow these steps to view your course progress:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Learning Center > All Courses .
3. On the Available Courses page, click the course for which you want to see your progress.
On the course page, the About this Course section displays your progress.
View Completed Courses
Log in to the Algorithm Lab and then, in the left navigation bar, click Learning Center > Completed to view your completed
courses.
Submit Reviews
You need to complete a course before you can submit a review on it.
Follow these steps to review courses:
1. Open the Completed Courses page.
2. Click the course for which you want to submit a review.
3. Scroll down to the User Reviews section and then click the number of stars you want to give the course in your review.
4. Write your review.
5. Click Submit Review .
The User Reviews section displays your review.
Report Errors
To report errors you experience with courses, email us an explanation of the error and the task URL where the error occurred.
Learning Center > Educators
Learning Center
Educators
Introduction
Educators are QuantConnect experts who contribute courses to the Learning Center. We are always looking for experts to
become Educators and share their insight with the community. As an Educator, you create the description, requirements, and
lessons of each course you contribute. You also provide the images for the course listing in the Learning Center. As students
complete your course, you'll receive course reviews so you can improve your course material.
Registration
Anyone can register to become an Educator. We are looking for Educators that specialize in the following areas:
Quant finance
HTML, CSS, and Javascript
C# and Python
Jupyter
If you have the skills listed above, contact us to start the registration process.
Compensation
Educators are compensated in knowledge, commissions, and exposure. You earn a $5,000 commission per accepted course
and a 70% revenue share for your courses. When your courses are listed in the Learning Center, your name and social media
accounts are public, giving you exposure to the QuantConnect community.
Learning Center > Course Structure
Learning Center
Course Structure
Introduction
The courses in the Learning Center are structured in a way so you can complete courses at your own pace. The course
structure enables you to improve your skills in finance, statistics, and software development while learning the QuantConnect
API in easily digestible portions. The idea behind the course lessons is to focus on implementing individual strategies rather than
learning just the theory.
Lessons
Courses are broken up into multiple lessons. Lessons are made up of videos, readings, and coding exercises. Lessons break up
the process of learning into digestible tasks. Each lesson builds on an understanding of the API from the lesson before it, so we
recommend completing the lessons in order. The introductory video of each lesson demonstrates the process of completing the
tasks in the lesson before you implement them yourself.
Tasks
Lessons are broken up into multiple tasks to test your understanding of the course topic. Each task is accompanied by text
instruction to guide you to complete the task. The Learning Center environment, where you complete each task, is similar to the
regular web IDE used for backtesting and live trading. After you read the task instructions and run your solution algorithm, you
are informed if you completed the task. If you need assistance, you can view a hint or the full solution file .
Results
To check if you have completed a task correctly, backtest your algorithm in the Learning Center environment and then the result
window displays your results. If you receive an error message, update the code and then run the backtest again. If you pass the
task, you'll be prompted to proceed to the next task in the course. If you're having trouble completing the task, you can copy the
task solution file.
Solutions
A solution file for each task is available in the Learning Center environment. You may use it, but we recommend you try to solve
the problem before you check the solution file. You'll learn best by solving the problem on your own. However, errors can occur
when running backtests, so you may use the solution file to ensure the environment is running without error.
Errors
If you run into an error when you are working on a task, compare your code to the solution file. The error may be caused by
either an error in your submission or an error in the backend of the Learning Center environment. In cases when errors occur in
the backtest of the Learning Center environment, email us the following information:
The URL of the task.
An explanation of the issue.
A code snippet to reproduce the error.
Projects
Projects
Projects contain files to run backtests, launch research notebooks, perform parameter optimizations, and deploy live trading
strategies. You need to create projects in order to create strategies and share your work with other members. Projects enable
you to generate institutional-grade reports on the performance of your backtests. You can create your projects from scratch or
you can utilize pre-built libraries and third-party libraries to expedite your development process.
Getting Started
Learn the basics
Structure
How projects are made
Files
Where code lives
IDE
A browser coding experience
Encryption
Another layer of IP protection
Debugging
Solve those coding errors
Collaboration
Work with your team members
Code Sessions
Connect to the remote IDE
Shared Libraries
Share code across projects
Package Environments
Bundled python libraries
LEAN Engine Versions
Customized LEAN engine versions
See Also
Backtesting
Sharing Backtests
Report
Projects > Getting Started
Projects
Getting Started
Introduction
Projects contain files to run backtests, launch research notebooks, perform parameter optimizations, and deploy live trading
strategies. You need to create projects in order to create strategies and share your work with other members. Projects enable
you to generate institutional-grade reports on the performance of your backtests. You can create your projects from scratch or
you can utilize pre-built libraries and third-party packages to expedite your development process.
The Algorithm lab enables you to create, store, and manage your projects in the cloud. You can only access your own projects
unless you share them with others, or add collaborators.
View All Projects
The All Projects page displays all of your QuantConnect projects in the organization, including libraries and Boot Camp lessons.
Click a project or directory on the page to open it.
Follow these steps to view the page:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Projects .
3. Click Open Project .
Create Projects
Follow these steps to create new projects:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Projects .
3. On the Projects page, click Create New Algorithm .
The web IDE displays an empty project.
Close Projects
In the Project panel, click Close to close projects.
Clone Projects
Clone a project to create a new copy of the project and save it within the same organization. When you clone a project, the
project files are duplicated but the backtest results and live deployment history are not retained. Cloning enables you to test
small changes in your projects before merging the changes back into the original project and start a new live deployment
record.
To clone a project, open the project and then, in the Project panel, click Clone .
Migrate Projects
Migrating moves a project from one organization to another. You must be the organization administrator to migrate projects out
of the organization. Migrate a project to run the project using resources from a different organization and to collaborate on the
project with members from a different organization. When you migrate projects, the project files are copied but the content
stored in the Object Store is not retained.
To migrate a project, open the project and then, in the Project panel, click Migrate .
Rename Projects
Follow these steps to rename a project:
1. Open the project .
2. In the Project panel, hover over the project name and then click the pencil icon that appears.
3. In the Name field, enter the new project name and then click Save Changes .
The project name must only contain - , _ , letters, numbers, and spaces. The project name can't start with a space or be any of
the following reserved names: CON, PRN, AUX, NUL, COM1, COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9, LPT1,
LPT2, LPT3, LPT4, LPT5, LPT6, LPT7, LPT8, or LPT9.
Create Project Directories
Set the name of a project to directoryName / projectName to create a project directory.
Set Descriptions
Follow these steps to set the project description:
1. Open the project .
2. In the Project panel, hover over the project name and then click the pencil icon that appears.
3. In the Description field, enter the new project description and then click Save Changes .
Edit Parameters
Algorithm parameters are hard-coded values for variables in your project that are set outside of the code files. Add parameters
to your projects to remove hard-coded values from your code files and to perform parameter optimizations. You can add
parameters, set default parameter values, and remove parameters from your projects.
Add Parameters
Follow these steps to add an algorithm parameter to a project:
1. Open the project .
2. In the Project panel, click Add New Parameter .
3. Enter the parameter name.
The parameter name must be unique in the project.
4. Enter the default value.
5. Click Create Parameter .
To get the parameter values into your algorithm, see Get Parameters .
Set Default Parameter Values
Follow these steps to set the default value of an algorithm parameter in a project:
1. Open the project .
2. In the Project panel, hover over the algorithm parameter and then click the pencil icon that appears.
3. Enter a default value for the parameter and then click Save .
The Project panel displays the default parameter value next to the parameter name.
Delete Parameters
Follow these steps to delete an algorithm parameter in a project:
1. Open the project .
2. In the Project panel, hover over the algorithm parameter and then click the trash can icon that appears.
3. Remove the GetParameter calls that were associated with the parameter from your code files.
Delete Projects
You can delete a project when it is open or closed.
Delete Open Projects
In the Project panel, click Delete , and then click Yes to delete the project.
Delete Closed Projects
Follow these steps to delete the project:
1. Open the My Projects page.
2. If the project is in a directory, click the directory files to navigate to the project file.
3. Hover over the project file and then click the trash can icon that appears.
"Project deleted" displays.
Encrypt Projects
When you save projects in QuantConnect Cloud, you can save encrypted versions of your project files instead of the raw,
human readable, file content. Encrypting your projects gives you an additional layer of protection. To use the encryption
system, you provide your own encryption key, which your local browser saves to memory. For more information about project
encryption, see Encryption .
Get Project Id
To get the project Id, open the project and check the URL. For example, the project Id of
https://www.quantconnect.com/project/13946911 is 13946911.
Projects > Structure
Projects
Structure
Introduction
Projects organize your algorithm data. They have settings, files, results, and attached libraries.
Your account has a directory to organize the projects that you have access to in each of your organizations. If you switch the
organization that you are connected as , your directory of projects is updated to reflect the projects that you have access to
within the new organization.
Files
New projects contain code files ( .py or .cs ) and notebook files ( .ipynb ). Run backtests with code files and launch the Research
Environment with notebook files. Code files must stay within your size quotas . To keep files small, files can import code from
other code files. To aid navigation, you can rename, move, and delete files in the web IDE. Notebook files save the input cells,
but not the output cells.
Directories
Your directory of projects can contain nested directories of projects to make navigation easier. Similarly, the code and notebook
files in your projects can contain nested directories of files. For example, if you have multiple Alpha models in your strategy, you
can create an alphas directory in your project to hold a file for each Alpha model.
The following directory names are reserved: .ipynb_checkpoints , .idea , .vscode , __pycache__ , bin , obj , backtests , live ,
optimizations , storage , and report .
Description
You can give a project a description to provide a high-level overview of the project and its functionality. Descriptions make it
easier to return to old projects and understand what is going on at a high level without having to look at the code. The project
description is also displayed at the top of backtest reports , which you can create after your backtest completes.
Libraries
Libraries are reusable code files that you can import into any project for use in backtesting, research, and live trading. Use
libraries to increase your development speed and save yourself from copy-pasting between projects. You can create libraries
and add them to your projects using the web IDE. Your libraries are saved under the Library directory in the Algorithm Lab.
Parameters
Algorithm parameters are hard-coded values for variables in your project that are set outside of the code files. Add parameters
to your projects to remove hard-coded values from your code files and to perform parameter optimizations . To get the
parameter values into your algorithm, see Get Parameters . The parameter values are sent to your algorithm when you deploy
the algorithm, so it's not possible to change the parameter values while the algorithm runs.
Projects > Files
Projects
Files
Introduction
The files in your projects enable you to implement trading algorithms, perform research, and store important information. Python
projects start with a main.py and a research.ipynb file. C# projects start with a Main.cs and a Research.ipynb file. Use the
main.py or Main.cs file to implement trading algorithms and use the ipynb file to access the Research Environment.
Supported File Types
The IDE supports the following file types:
.cs
.ipynb
.py
.html
.css
Add Files
Follow these steps to add a file to a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, expand the Workspace (Workspace) section.
4. Click the New File icon.
5. Enter a file name and extension.
6. Press Enter .
Add Directories
Follow these steps to add a directory to a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, expand the Workspace (Workspace) section.
4. Click the New Directory icon.
5. Enter a directory name and then press Enter .
The following directory names are reserved: .ipynb_checkpoints , .idea , .vscode , __pycache__ , bin , obj , backtests , live ,
optimizations , storage , and report .
Open Files
Follow these steps to open a file in a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, click the file you want to open.
Close Files
To close a file, at the top of the IDE, click the x button on the file tab you want to close.
To close all of the files in a project, at the top of the IDE, right-click one of the file tabs and then click Close All .
Rename Files and Directories
Follow these steps to rename a file or directory in a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, right-click the file or directory you want to rename and then click Rename .
4. Enter the new name and then press Enter .
The following directory names are reserved: .ipynb_checkpoints , .idea , .vscode , __pycache__ , bin , obj , backtests , live ,
optimizations , storage , and report .
Delete Files and Directories
Follow these steps to delete a file or directory in a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, right-click the file or directory you want to delete and then click Delete Permanently .
4. Click Delete .
Size Quotas
The maximum file size you can have in a project depends on your organization's tier. The following table shows the quota of
each tier:
Tier Max File Size (KB)
Free 32
Quant Researcher 64
Team 128
Trading Firm 256
Institution 256
Projects > IDE
Projects
IDE
Introduction
The web Integrated Development Environment (IDE) lets you work on research notebooks and develop algorithms for
backtesting and live trading. When you open a project , the IDE automatically displays. You can access your trading algorithms
from anywhere in the world with just an internet connection and a browser. If you prefer to use a different IDE, the CLI allows
you to develop locally in your preferred IDE.
Supported Languages
The Lean engine supports C# and Python. Python is less verbose, has more third-party libraries, and is more popular among the
QuantConnect community than C#. C# is faster than Python and it's easier to contribute to Lean if you have features written in
C# modules. Python is also the native language for the research notebooks, so it's easier to use in the Research Environment.
The programming language that you have set on your account determines how autocomplete and IntelliSense are verified and
determines the types of files that are included in your new projects. If you have Python set as your programming language, new
projects will have .py files. If you have C# set as your programming language, new projects will have .cs files.
Change Languages
Follow these steps to select a programming language:
1. Log in to the Algorithm Lab.
2. In the top navigation bar, click yourUsername > My Account .
3. On your Account page, in the Account Settings section, click C# or Python .
"Preferred language setting has been updated" displays.
Autocomplete and Intellisense
Intellisense is a GUI tool in your code files that shows auto-completion options and presents the members that are accessible
from the current object. The tool works by searching for the statement that you're typing, given the context. You can use
Intellisense to auto-complete method names and object attributes. When you use it, a pop-up displays in the IDE with the
following information:
Member type
Member description
The parameters that the method accepts (if the member is a method)
Use Intellisense to speed up your algorithm development. It works with all of the default class members in Lean, but it doesn't
currently support class names or user-defined objects.
Use Autocomplete
Follow these steps to use autocomplete:
1. Open a project .
2. Type the first few characters of a variable, function, class, or class member that you want to autocomplete (for example,
self.set or SimpleMovingAverage.Upda ).
3. Press CTRL+Space .
If there are class members that match the characters you provided, a list of class members displays.
4. Select the class member that you want to autocomplete.
The rest of the class member name is automatically written in the code file.
Console
The console panel at the bottom of the IDE provides some helpful information while you're developing algorithms.
Cloud Terminal
The Cloud Terminal tab of the panel shows the API messages, errors, and the logs from your algorithms.
To clear the Cloud Terminal, click the Clear Logs icon in the top-right corner of the panel.
Ask Mia
The Ask Mia tab of the panel is where you can interact with our AI assistant, Mia.
Mia provides contextual assistance to most issues you may encounter when developing a strategy, including build errors, API
methods, and best coding practices. It has been trained on hundreds of algorithms and thousands of documentation pages.
To clear the chat with Mia, click the Clear Mia Chat icon in the top-right corner of the panel.
Problems
The Problems tab of the panel highlights the coding errors in your algorithms.
Manage Nodes
The Resources panel shows the cloud backtesting, research, and live trading nodes within your organization.
To view the Resources panel, open a project and then, in the right navigation menu, click the Resources icon.
The panel displays the following information for each node:
Column Description
Node The node name and model.
In Use By The owner and name of the project using the node.
To stop a running node, click the stop button next to it. You can stop nodes that you are using, but you need stop node
permissions to stop nodes other members are using.
By default, we select the best node available in your clusters when you launch a backtest or research notebook. To use a
specific node, click the check box next to a node in the panel.
Navigate the File Outline
The Outline section in the Explorer panel is an easy way to navigate your files. The section shows the name of classes,
members, and functions defined throughout the file. Click one of the names to jump your cursor to the respective definition in
the file. To view the Outline , open a project and then, in the right navigation menu, click the Explorer icon.
Split the Editor
The editor can split horizontally and vertically to display multiple files at once. Follow these steps to split the editor:
1. Open a project .
2. In the right navigation bar, click the Explorer icon.
3. In the QC (Workspace) section, drag and drop the files you want to open.
Use this feature instead of opening multiple browser tabs for a single project. If you open open multiple browser tabs, two code
sessions will be updating the same project, which will cause the code sessions to fall out of sync.
Show and Hide Code Blocks
The editor can hide and show code blocks to make navigating files easier. To hide and show code blocks, open a project and
then click the arrow icon next to a line number.
Keyboard Shortcuts
Keyboard shortcuts are combinations of keys that you can issue to manipulate the IDE. They can speed up your workflow
because they remove the need for you to reach for your mouse.
Follow these steps to view the keyboard shortcuts of your account:
1. Open a project .
2. Press F1 .
3. Enter "Preferences: Open Keyboard Shortcuts".
4. Click Preferences: Open Keyboard Shortcuts .
To set a key binding for a command, click the pencil icon in the left column of the keyboard shortcuts table, enter the key
combination, and then press Enter .
Themes
The Algorithm Lab offers light and dark themes. Follow these steps to change themes:
1. Log in to your account.
2. In the top navigation bar, click yourUsername > My Account .
3. On your Account page, in the Account Settings section, click Light Theme or Dark Theme
Your Account page refreshes and displays the new theme.
Supported Browsers
The IDE works with Chrome, Edge, Firefox, and Safari. For more information about browser support, see Browser Support in the
Visual Studio Code documentation.
Cookies
Cookies are essential for the Algorithm Lab to connect to the VS Code environment because modern browsers treat the coding
environment domain (e.g., {...}.code.qc.com) as a third-party domain relative to the main domain (www.qc.com). This
distinction means that if cookies for the coding environment are blocked, essential functions like user authentication, passing
messages, adjusting panel sizes, setting tasks in bootcamp, and encrypting/decrypting project files will not work correctly,
leading to a disrupted user experience. Enabling cookies ensures that your coding session remains seamless and fully
functional. To enable third-party cookies, see the support page of the following browsers:
Chrome
Edge
Firefox
Safari
If you use Safari, disable "Prevent cross-site tracking" .
If you use Chrome, add [*.]quantconnect.com under "Sites allowed to use third-party cookies".
Troubleshooting
If you experience issues trying to load the IDE, follow these steps:
1. Check if you're using one of the supported browsers .
2. Check your internet connection and speed.
3. Test a different supported browser.
4. Enable cookies .
5. Disable your Virtual Private Network (VPN).
6. Disable your browser add-ons.
7. Check your anti-virus settings.
The internet protection of some anti-virus products block the "Service Workers" that the IDE needs to operate. Kaspersky
and Avast are the two products that commonly block the IDE from using Service Workers. These are background threads
that improve the IDE's experience.
8. Configure your network settings to use Google Public DNS .
The DNS settings of some ISPs block Microsoft DNS, so some panels don't load and display the following message:
"Server IP address could not be found".
The following resources explain how to change your networks settings:
Firefox DNS-over-HTTPS
DNS Encryption with DNS over HTTPs (DoH) on Chrome
Microsoft Edge to have DNS over HTTPS (DoH) as the default DNS settings
Set up 1.1.1.1 - macOS
9. Try to load the IDE with a different computer, tablet, or cell phone.
10. Clear your browser cache, especially if you created your project before the new IDE (March 2022).
Your browser may cache data from the old IDE.
11. Check if your hard drive is full.
12. Restart your computer and internet router.
If these steps don't solve the issues, send the logs to the Console Inspector to the Support Team .
Projects > Encryption
Projects
Encryption
Introduction
When you save projects in QuantConnect Cloud, you can save encrypted versions of your project files instead of the raw,
human readable, file content. To use the encryption system, you provide your own encryption key, which your local browser
saves to memory. Afterwards, whenever you save your project, your browser uses the local key to encrypt your project files and
then only the obfuscated files are pushed to QuantConnect Cloud.
View All Keys
The organization homepage displays the name of all your organization's encryption keys and provides a way to upload more. A
MD5 hash of the key is recorded to know which key encoded a project.
The icon means the key is registered in the database but we could not locate it on your computer. This may happen when
you login from a new computer. To use this key, click Use Key and upload the key with the pop-up. This process stores a copy
of your key in the browser local store so you can decrypt the relevant projects in your organization.
Add Keys
Follow these steps to add an encryption key to your local storage:
1. Navigate to the Encryption Keys section of your organization homepage.
2. Click Add Encryption Key .
3. In the Add Encryption Key window, enter a name for the key and then add the encryption key file.
The encryption key file must be a txt file with at least 32 characters. Itʼs content can be arbitrary.
4. Click Add Key .
Delete Keys
Follow these steps to delete an encryption key:
1. Navigate to the Encryption Keys section of your organization homepage.
2. Click Delete next to the key you want to delete.
3. Click Delete .
Add Encryption
Follow these steps to add encryption to your project files:
1. Open a project or create a new one .
2. In the Project panel, click Encrypt Project .
3. If you have added an encryption key before and want to use it for this project, click the key name from the drop-down
menu.
4. If you havenʼt added an encryption key before or you want to add a new one, follow these steps:
1. Click Add Key to Organization from the drop-down menu.
2. In the Add Encryption Key window, enter a name for the key and then add the encryption key file.
The encryption key file must be a txt file with at least 32 characters. Itʼs content can be arbitrary.
3. Click Add Key .
5. Click Encrypt .
Remove Encryption
To remove encryption from a project, open the project and then, in the Project panel, click the unlock icon next to the name of
your encryption key.
Collaboration Support
Encryption isnʼt available for projects that have collaborators .
Libraries
Encrypted projects can use libraries encrypted with the same project key or unencrypted libraries. However, you cannot use a
library encrypted with a different project encryption key.
To encrypt a library, open its project and set its encryption key .
Projects > Debugging
Projects
Debugging
Introduction
Debugging is the process of systematically using a tool to find and fix errors in software. Errors can cause unintended trades,
unexpected algorithm crashes, and faulty risk management logic. We use a debugging tool step through code line-by-line, and
inspect the variables to understand the internal state of the program. You have many tools to debug your algorithm, including
our built-in debugger, logging statements, charting, and the Object Store.
Coding Errors
Coding errors are errors that cause your projects to not build, throw exceptions, or behave unexpectedly. There are generally 3
types of coding errors: build, runtime, and logic errors. Each type of error occurs for different reasons.
Build Errors
Build errors occur when the interpreter's syntax check fails. An example code snippet that produces a build error is the
following:
If build errors occur in your project, you can not use the debugger, logging statements, or custom charts to debug the issue.
You're notified of build errors in the following ways:
The line where the error occurs is underlined in red.
The Problems panel at the bottom of the IDE displays the errors.
The Explorer panel highlights the editors, files, and outlines in red where the error occurs.
Runtime Errors
Runtime Errors, also called exceptions, occur when the interpreterʼs syntax checks pass but an error occurs during execution.
An example code snippet that produces a runtime error is the following:
If runtime errors occur in your project, a stack trace of the error is added to the Cloud Terminal and the log file. For example, the
snippet above produces the following error message:
Runtime Error: IndexError : list index out of range
at OnData
a[1]
===
at Python.Runtime.PyObject.Invoke(PyTuple args in main.py: line 17
(Open Stack Trace)
The stack trace from the build error identifies the line of code where the error occurs. If the error doesn't reference your project
a = 1
if a = 2:
pass
a = [1]
a[1]
PY
PY
files, it's an issue with Lean or another library. To view more information about the error, click (Open Stack Trace) .
Logic Errors
Logic errors occur when your algorithm behaves in an unexpected or unintended manner. These types of errors don't halt the
program execution, so they are difficult to diagnose. An example code snippet that produces a logic error is the following:
To resolve logic errors, carefully trace your algorithm. You may use the log method and debug method methods or the built-in
debugger.
Debugger
The debugger is a built-in tool to help you debug coding errors while backtesting. The debugger enables you to slow down the
code execution, step through the program line-by-line, and inspect the variables to understand the internal state of the program.
For more information about the backtesting debugger, see Backtest Debugging .
Logging Statements
Algorithms can record string messages ('log statements') to a file for analysis after a backtest is complete, or as a live algorithm
is running. These records can assist in debugging logical flow errors in the project code. Consider adding them in the code block
of an if statement to signify an error has been caught.
It's good practice to add logging statements to live algorithms so you can understand its behavior and keep records to compare
against backtest results. If you don't add logging statements to a live algorithm and the algorithm doesn't trade as you expect,
it's difficult to evaluate the underlying problem.
Log
Log statements are added to the log file while your algorithm continues executing. Logging dataset information is not permitted.
Use log method statements to debug your backtests and live trading algorithms.
Log length is capped by organization tier . If your organization hits the daily limit, contact us .
If you log the same content multiple times, only the first instance is added to the log file. To bypass this rate-limit, add a
timestamp to your log messages.
For live trading, the log files of each cloud project can store up to 100,000 lines for up to one year. If you log more than 100,000
lines or some lines become older than one year, we remove the oldest lines in the files so your project stays within the quota.
To record the algorithm state when the algorithm stops executing, add log statements to the on_end_of_algorithm event
handler.
Debug
Debug statements are the same as log statements, but debug method statements are orange in the Cloud Terminal. Use these
statements when you want to give more attention to a message in the Cloud Terminal. Debug messages can be up to 200
characters in length. If you send multiple debug statements within 1 second, your messages are rate-limited to avoid crashing
average = x + y / 2 # instead of (x + y) / 2
self.log("My log message")
PY
PY
your browser.
Error
Error statements are the same as log statements, but error method statements are displayed in red text in the Cloud Terminal.
Use these statements when you want to give the most attention to a message in the Cloud Terminal. Error statements are ratelimited like debug statements.
Quit
Quit statements cause your project to stop running and may log some data to the log file and Cloud Terminal. These statements
are orange in the Cloud Terminal. When you call the quit method method, the program continues executing until the end of the
method definition. If you want to quit execution immediately, return after you call quit method.
Charting
You can use the IDE charting capabilities to plot values over time when debugging. To add data points to a custom chart, call the
plot method with a chart name, series name, and value. For a full example, see Charting .
If you run your algorithm in QuantConnect Cloud, we limit the number of points a chart can have to 4,000 because intensive
charting generates hundreds of megabytes (200MB) of data, which is too much to stream online or display in a web browser. If
you exceed the limit, the following error message is thrown:
Exceeded maximum data points per series, chart update skipped.
Object Store
The Object Store is a key-value data store for low-latency information storage and retrieval. During a backtest, you can build
large objects youʼd like to analyze and write them for later analysis. This workflow can be helpful when the objects are large and
plotting is impossible or when you want to perform analysis across many backtests.
For more information about the Object Store, see Object Store . For a specific example of saving indicator values during a
backtest into the Object Store and then plotting them in the Research Environment, see Example for Plotting .
self.debug("My debug message")
self.error("My error message")
self.quit("My quit message")
# Add data points to a custom chart.
self.plot("Chart Name", "Series Name", value)
# Save the key-value pair in the Object Store.
self.object_store.save("key", value)
PY
PY
PY
PY
PY
Projects > Collaboration
Projects
Collaboration
Introduction
Project collaboration is a real-time coding experience with other members of your team. Collaborating can speed up your
development time. By working with other members in an organization, members within the organization can specialize in
different parts of the project.
Video Demo
When there are multiple people working on the same project, the cursor of each member is visible in the IDE and all file changes
occur in real-time for everyone. The following video demonstrates the collaboration feature:
Add Team Members
You need to own the project to add team members to it.
Follow these steps to add team members to a project:
1. Open the project .
2. In the Collaborate section of the Project panel, click Add Collaborator .
3. Click the Select User... field and then click a member from the drop-down menu.
4. If you want to give the member control of the project's live deployments , select the Live Control check box.
5. Click Add User .
The member you add receives an email with a link to the project.
If the project has a shared library , the collaborator can access the project, but not the library. To grant them access to the
library, add them as a collaborator to the library project.
Collaborator Quotas
The number of members you can add to a project depends on your organization's tier . The following table shows the number of
collaborators each tier can have per project:
Tier Collaborators per Project
Free Unsupported
Quant Researcher Unsupported
Team 10
Trading Firm Unlimited
Institution Unlimited
Toggle Live Control
You need to have added a member to the project to toggle their live control of the project.
Follow these steps to enable and disable live control for a team member:
1. Open the project .
2. In the Collaborate section of the Project panel, click the profile image of the team member.
3. Click the Live Control check box.
4. Click Save Changes .
Remove Team Members
Follow these steps to remove a team member from a project you own:
1. Open the project .
2. In the Collaborate section of the Project panel, click the profile image of the team member.
3. Click Remove User .
To remove yourself as a collaborator from a project you don't own, delete the project .
Intellectual Property
All individuals on QuantConnect own their intellectual property (IP) on our platform. Your code is private and only accessible by
people you share the project with and with support-engineers when you submit a support ticket. At no point does QuantConnect
ever claim ownership of user IP. The only case where algorithm code becomes public domain is when they are shared to the
forum. In this case, algorithms need to be made public domain to allow the sharing of the algorithm code.
It is common when companies hire engineers to write software, they require their employees to sign an agreement that gives the
company IP ownership of any code written for work. They need this because they're paying you to write software, and the
company needs to sell that software to turn a profit. Similarly, the Organizations feature allows you to control who holds IP
ownership over a project. Each type of organization has its own mechanisms for handling project IP ownership.
Individual Organizations
The Free and the Quant Researcher tiers only allow single-member organizations. This means you can't collaborate with anyone
else inside the QuantConnect platform. Simply put, you own the IP for any projects you work on since you are the sole
collaborator.
Team Organizations
For organizations that allow multiple users to collaborate on projects, the user who created the project owns it; this can be you
or one of your teammates. If you add a teammate/collaborate, they can clone it, but the original project belongs to the person
who first created it.
Trading Firm & Institution Organizations
For Trading Firm and Institution organizations, which are generally used by companies and funds, the firm owns all employee
projects. This is made to suit firms that wish to hire consultants and need to ensure the code remains with the company when
the consultant work is finished. You have to explicitly create a project in an organization for it to be created on the
organization's account.
Other Collaboration Methods
Additional methods of collaboration include cloning, sharing, and migrating projects.
Clone Projects
Clone a project to create a new copy of the project and save it within the same organization. When you clone a project, the
project files are duplicated but the backtest results and live deployment history are not retained. Cloning enables you to test
small changes in your projects before merging the changes back into the original project and start a new live deployment
record.
To clone projects, open the project you want to clone and then, in the Project panel, click Clone . "Project cloned successfully"
displays.
Share Projects
Run a backtest and then make the backtest results public to share a project. Once a backtest is made public, a link is generated
for you that opens the backtest results and the project files. You can directly give the link to others, attach the backtest to a
forum discussion, or embed the backtest into a website. However, note that when you make a backtest public, the project files
are accessible to anyone who visits the link, even after you delete the project. As a result, we don't recommend collaborating on
projects by making backtests public and sharing the link with your collaborators. Instead, add team members to your project
since it protects your intellectual property.
You can share a backtest at any time when it's executing. Although, if you generate a link to share the backtest before the
backtest completes, the link that's generated will not contain all of the backtest results. Some reasons to share your project
include the following:
Attach the project to the forum to ask for help, gather feedback, or report an issue.
Attach the project to a data issue to reduce the amount of time it takes to fix the data issue.
Share a link to the project with others to give them a copy of the project files and the backtest results.
To share a research notebook, save the notebook and run a backtest.
Migrate Projects
Migrating moves a project from one organization to another. You must be the organization administrator to migrate projects out
of the organization. Migrate a project to run the project using resources from a different organization and to collaborate on the
project with members from a different organization. When you migrate projects, the project files are copied but the content
stored in the Object Store is not retained.
Follow these steps to migrate projects:
1. Open the project you want to migrate.
2. In the Project panel, click Migrate .
3. Click the name of the organization to which you want to migrate the project and then click Migrate .
The top navigation bar displays Connected as: theOrganizationYouMigratedTo .
Projects > Code Sessions
Projects
Code Sessions
Introduction
Code sessions let you access a cloud hosted IDE to research and develop trading algorithms. When you open a project, a new
code session starts running with the latest master branch of the LEAN trading engine. The session is ready to go with access to
the full QuantConnect data library and the cloud resources of the QuantConnect technology stack.
View Code Sessions
The Projects page displays all of your running code sessions in your current organization. To view the page, log in to the
Algorithm Lab and then, in the left navigation menu, click Projects .
To open one of the code sessions, click the project name.
To stop the code sessions, click the stop icon next to a project name. If you log out, the code sessions don't automatically stop.
The left navigation bar of the Algorithm Lab also shows the running code sessions underneath Projects . The blue code session
represents the session that's currently open. The gray code sessions represent the sessions that are currently minimized.
Code Session Quotas
If you have a project open, it uses a coding session. Paid organizations can have multiple active coding sessions, but free users
can only have one coding session open at a time. The following table shows how many active coding sessions you can have on
each organization tier:
Tier Initial Coding Session Quota
Quant Researcher 2
Team 4
Trading Firm 8
Institution 16
If the organization you're in has more live trading nodes than your initial coding session quota, then your coding session quota
increases to the number of live trading nodes you have in the organization so you can view all your live strategies.
The quota for free organizations is a global quota, so you can have one active coding session across all of your free
organizations. The quotas for paid organizations are at the organization level. Therefore, if you are in two Quant Researcher
organizations, you can have two active coding sessions in one of those organizations and another two active sessions in the
other organization. These paid tier quotas are for each account, not for the organization as a whole. For instance, a Trading Firm
organization can have more than eight members and all of the members can simultaneously work on projects within the
organization.
Projects > Shared Libraries
Projects
Shared Libraries
Introduction
Project libraries are QuantConnect projects you can merge into your project to avoid duplicating code files. If you have tools that
you use across several projects, create a library.
Create Libraries
Follow these steps to create a library:
1. Create a new project .
2. In the project panel, click Add Library .
3. Click Create New .
4. In the Input Library Name field, enter a name for the library (for example, Calculators ).
To create a library directory, set the name to directoryName / libraryName (for example, Tools / Calculators ).
5. Click Create Library .
The template library files are added to your project. View the files in the Explorer panel.
6. In the right navigation menu, click the Explorer icon.
7. In Explorer panel, open the Library.py file, rename it to reflect its purpose (e.g.: TaxesCalculator.py ), and implement your
library.
Add Libraries
Follow these steps to add a library to your project:
1. Open the project .
2. In the Project panel, click Add Library .
3. Click the Choose a library... field and then click a library from the drop-down menu.
4. Click Add Library (e.g. Calculators ).
The library files are added to your project. To view the files, in the right navigation menu, click the Explorer icon.
5. Import the library into your project to use the library.
Rename Libraries
To rename a library, open the library project file and then rename the project .
Remove Libraries
Follow these steps to remove a library from your project:
from Calculators.TaxesCalculator import TaxesCalculator
class AddLibraryAlgorithm(QCAlgorithm):
taxes_calculator = TaxesCalculator()
PY
1. Open the project that contains the library you want to remove.
2. In the Project panel, hover over the library name and then click the trash can icon that appears.
The library files are removed from your project.
Delete Libraries
To delete a library, delete the library project file .
Projects > Package Environments
Projects
Package Environments
Introduction
Libraries (or packages) are third-party software that you can use in your projects. You can use many of the available opensource libraries to complement the classes and methods that you create. Libraries reduce your development time because it's
faster to use a pre-built, open-source library than to write the functionality. Libraries can be used in backtesting, research, and
live trading. The environments support various libraries for machine learning, plotting, and data processing. As members often
request new libraries, we frequently add new libraries to the underlying docker image that runs the Lean engine.
This feature is primarily for Python algorithms as not all Python libraries are compatible with each other. We've bundled together
different sets of libraries into distinct environments. To use the libraries of an environment, set the environment in your project
and add the relevant import statement of a library at the top of your file.
Set Environment
Follow these steps to set the library environment:
1. Open a project .
2. In the Project panel, click the Python Foundation field and then select an environment from the drop-down menu.
Default Environment
The default environment supports the following libraries:
absl-py 2.1.0
accelerate 0.34.2
adagio 0.2.6
aesara 2.9.4
aiohappyeyeballs 2.4.4
aiohttp 3.11.10
aiosignal 1.3.1
aiosqlite 0.20.0
alembic 1.14.0
alibi-detect 0.12.0
alphalens-reloaded 0.4.5
altair 5.5.0
anaconda-anon-usage 0.4.4
annotated-types 0.7.0
anyio 4.7.0
aplr 10.8.0
appdirs 1.4.4
apricot-select 0.6.1
arch 7.2.0
archspec 0.2.3
argon2-cffi 23.1.0
argon2-cffi-bindings 21.2.0
arrow 1.3.0
arviz 0.20.0
astropy 7.0.0
astropy-iers-data 0.2024.12.9.0.36.21
asttokens 3.0.0
astunparse 1.6.3
async-lru 2.0.4
attrs 24.2.0
Authlib 1.3.2
autograd 1.7.0
autograd-gamma 0.5.0
autokeras 2.0.0
autoray 0.7.0
PY
a
u
t
o
r
a
y
0.7.0
a
x
-
p
l
a
t
f
o
r
m
0.4.3
b
a
b
e
l
2.1
6.0
b
a
y
e
s
i
a
n
-
o
p
t
i
m
i
z
a
t
i
o
n
2.0.0
b
e
a
u
t
i
f
u
l
s
o
u
p
4
4.1
2.3
b
l
e
a
c
h
6.2.0
b
l
i
n
k
e
r
1.9.0
b
l
i
s
0.7.1
1
b
l
o
s
c
2
2.7.1
b
o
k
e
h
3.6.2
b
o
l
t
o
n
s
2
3.0.0
b
o
t
o
r
c
h
0.1
2.0
B
o
t
t
l
e
n
e
c
k
1.4.2
B
r
o
t
l
i
1.0.9
c
a
c
h
e
t
o
o
l
s
5.5.0
c
a
p
t
u
m
0.7.0
c
a
t
a
l
o
g
u
e
2.0.1
0
c
a
t
b
o
o
s
t
1.2.7
c
a
t
e
g
o
r
y
-
e
n
c
o
d
e
r
s
2.6.4
c
a
u
s
a
l
-
c
o
n
v
1
d
1.5.0.p
o
s
t
8
c
e
r
t
i
f
i
2
0
2
4.8.3
0
c
e
s
i
u
m
0.1
2.1
c
f
f
i
1.1
7.1
c
h
a
r
d
e
t
5.2.0
c
h
a
r
s
e
t
-
n
o
r
m
a
l
i
z
e
r
3.3.2
c
h
e
c
k
-
s
h
a
p
e
s
1.1.1
c
h
r
o
n
o
s
-
f
o
r
e
c
a
s
t
i
n
g
1.4.1
c
l
a
r
a
b
e
l
0.9.0
c
l
i
c
k
8.1.7
c
l
i
k
i
t
0.6.2
c
l
o
u
d
p
a
t
h
l
i
b
0.2
0.0
c
l
o
u
d
p
i
c
k
l
e
3.1.0
c
m
d
s
t
a
n
p
y
1.2.4
c
o
l
o
r
a
m
a
0.4.6
c
o
l
o
r
c
e
t
3.1.0
c
o
l
o
r
l
o
g
6.9.0
c
o
l
o
r
l
o
v
e
r
0.3.0
c
o
l
o
u
r
0.1.5
c
o
m
m
0.2.2
c
o
n
d
a
2
4.9.2
c
o
n
d
a
-
c
o
n
t
e
n
t
-
t
r
u
s
t
0.2.0
c
o
n
d
a
-
l
i
b
m
a
m
b
a
-
s
o
l
v
e
r
2
4.9.0
c
o
n
d
a
-
p
a
c
k
a
g
e
-
h
a
n
d
l
i
n
g
2.3.0
c
o
n
d
a
_
p
a
c
k
a
g
e
_
s
t
r
e
a
m
i
n
g
0.1
0.0
c
o
n
f
e
c
t
i
o
n
0.1.5
c
o
n
s
0.4.6
c
o
n
t
o
u
r
p
y
1.3.1
c
o
n
t
r
o
l
0.1
0.1
c
o
p
u
l
a
e
0.7.9
c
o
p
u
l
a
s
0.1
2.0
c
o
r
e
f
o
r
e
c
a
s
t
0.0.1
5
c
r
a
m
j
a
m
2.9.0
c
r
a
s
h
t
e
s
t
0.3.1
c
r
e
m
e
0.6.1
c
r
y
p
t
o
g
r
a
p
h
y
4
3.0.0
c
u
c
i
m
-
c
u
1
2
2
4.8.0
c
u
d
a
-
p
y
t
h
o
n
1
2.6.2.p
o
s
t
1
c
u
d
f
-
c
u
1
2
2
4.8.3
c
u
f
f
l
i
n
k
s
0.1
7.3
c
u
g
r
a
p
h
-
c
u
1
2
2
4.8.0
c
u
m
l
-
c
u
1
2
2
4.8.0
c
u
p
r
o
j
-
c
u
1
2
2
4.8.0
c
u
p
y
-
c
u
d
a
1
2
x
1
3.3.0
c
u
s
p
a
t
i
a
l
-
c
u
1
2
2
4.8.0
c
u
v
s
-
c
u
1
2
2
4.8.0
c
u
x
f
i
l
t
e
r
-
c
u
1
2
2
4.8.0
c
v
x
o
p
t
1.3.2
c
v
x
p
o
r
t
f
o
l
i
o
1.4.0
c
v
x
p
y
1.6.0
c
y
c
l
e
r
0.1
2.1
c
y
m
e
m
2.0.1
0
C
y
t
h
o
n
3.0.1
1
d
a
r
t
s
0.3
1.0
d
a
s
h
2.9.3
d
a
s
h
-
c
o
r
e
-
c
o
m
p
o
n
e
n
t
s
2.0.0
d
a
s
h
_
c
y
t
o
s
c
a
p
e
1.0.2
d
a
s
h
-
h
t
m
l
-
c
o
m
p
o
n
e
n
t
s
2.0.0
d
a
s
h
-
t
a
b
l
e
5.0.0
d
a
s
k
2
0
2
4.7.1
d
a
s
k
-
c
u
d
a
2
4.8.2
d
a
s
k
-
c
u
d
f
-
c
u
1
2
2
4.8.3
d
a
s
k
-
e
x
p
r
1.1.9
d
a
s
k
-
e
x
p
r
1.1.9
d
a
t
a
b
r
i
c
k
s
-
s
d
k
0.3
8.0
d
a
t
a
c
l
a
s
s
e
s
-
j
s
o
n
0.6.7
d
a
t
a
s
e
t
s
2.2
1.0
d
a
t
a
s
h
a
d
e
r
0.1
6.3
d
e
a
p
1.4.1
d
e
b
u
g
p
y
1.8.9
d
e
c
o
r
a
t
o
r
5.1.1
d
e
e
p
m
e
r
g
e
2.0
d
e
f
u
s
e
d
x
m
l
0.7.1
D
e
p
r
e
c
a
t
e
d
1.2.1
5
d
e
p
r
e
c
a
t
i
o
n
2.1.0
d
g
l
2.1.0
d
i
l
l
0.3.8
d
i
m
o
d
0.1
2.1
7
d
i
r
t
y
j
s
o
n
1.0.8
d
i
s
k
c
a
c
h
e
5.6.3
d
i
s
t
r
i
b
u
t
e
d
2
0
2
4.7.1
d
i
s
t
r
i
b
u
t
e
d
-
u
c
x
x
-
c
u
1
2
0.3
9.1
d
i
s
t
r
o
1.9.0
d
m
-
t
r
e
e
0.1.8
d
o
c
k
e
r
7.1.0
d
o
c
u
t
i
l
s
0.2
1.2
D
o
u
b
l
e
M
L
0.9.0
d
r
o
p
s
t
a
c
k
f
r
a
m
e
0.1.1
d
t
r
e
e
v
i
z
2.2.2
d
t
w
-
p
y
t
h
o
n
1.5.3
d
w
a
v
e
-
c
l
o
u
d
-
c
l
i
e
n
t
0.1
3.1
d
w
a
v
e
-
d
r
i
v
e
r
s
0.4.4
d
w
a
v
e
-
g
a
t
e
0.3.2
d
w
a
v
e
-
g
r
e
e
d
y
0.3.0
d
w
a
v
e
-
h
y
b
r
i
d
0.6.1
2
d
w
a
v
e
-
i
n
s
p
e
c
t
o
r
0.5.1
d
w
a
v
e
-
i
n
s
p
e
c
t
o
r
a
p
p
0.3.3
d
w
a
v
e
-
n
e
a
l
0.6.0
d
w
a
v
e
_
n
e
t
w
o
r
k
x
0.8.1
5
d
w
a
v
e
-
o
c
e
a
n
-
s
d
k
8.0.1
d
w
a
v
e
-
o
p
t
i
m
i
z
a
t
i
o
n
0.3.0
d
w
a
v
e
-
p
r
e
p
r
o
c
e
s
s
i
n
g
0.6.6
d
w
a
v
e
-
s
a
m
p
l
e
r
s
1.3.0
d
w
a
v
e
-
s
y
s
t
e
m
1.2
6.0
d
w
a
v
e
-
t
a
b
u
0.5.0
d
w
a
v
e
b
i
n
a
r
y
c
s
p
0.3.0
e
c
o
s
2.0.1
4
e
i
n
o
p
s
0.8.0
e
i
n
x
0.3.0
E
M
D
-
s
i
g
n
a
l
1.6.4
e
m
p
y
r
i
c
a
l
-
r
e
l
o
a
d
e
d
0.5.1
1
e
n
-
c
o
r
e
-
w
e
b
-
m
d
3.7.1
e
n
-
c
o
r
e
-
w
e
b
-
s
m
3.7.1
e
t
_
x
m
l
f
i
l
e
2.0.0
e
t
u
p
l
e
s
0.3.9
e
x
c
h
a
n
g
e
_
c
a
l
e
n
d
a
r
s
4.6
e
x
e
c
u
t
i
n
g
2.1.0
f
a
i
s
s
-
c
p
u
1.9.0.p
o
s
t
1
F
a
r
a
m
a
-
N
o
t
i
f
i
c
a
t
i
o
n
s
0.0.4
f
a
s
t
a
i
2.7.1
8
f
a
s
t
a
i
2
0.0.3
0
f
a
s
t
c
o
r
e
1.7.2
6
f
a
s
t
d
o
w
n
l
o
a
d
0.0.7
f
a
s
t
e
n
e
r
s
0.1
9
f
a
s
t
j
s
o
n
s
c
h
e
m
a
2.2
1.1
f
a
s
t
p
a
r
q
u
e
t
2
0
2
4.1
1.0
f
a
s
t
p
r
o
g
r
e
s
s
1.0.3
f
a
s
t
r
l
o
c
k
0.8.2
f
a
s
t
t
e
x
t
0.9.3
f
e
a
t
u
r
e
-
e
n
g
i
n
e
1.6.2
f
e
a
t
u
r
e
t
o
o
l
s
1.3
1.0
f
i
l
e
l
o
c
k
3.1
6.1
f
i
l
e
t
y
p
e
1.2.0
f
i
n
d
i
f
f
0.1
0.2
F
i
x
e
d
E
f
f
e
c
t
M
o
d
e
l
0.0.5
F
l
a
g
E
m
b
e
d
d
i
n
g
1.2.1
1
F
l
a
s
k
3.1.0
f
l
a
t
b
u
f
f
e
r
s
2
4.3.2
5
f
o
n
t
t
o
o
l
s
4.5
5.2
f
o
r
m
u
l
a
i
c
1.0.2
f
q
d
n
1.5.1
f
r
o
z
e
n
d
i
c
t
2.4.2
f
r
o
z
e
n
l
i
s
t
1.5.0
f
s
2.4.1
6
f
s
s
p
e
c
2
0
2
4.6.1
f
s
s
p
e
c
2
0
2
4.6.1
f
u
g
u
e
0.9.1
f
u
t
u
r
e
1.0.0
f
u
z
z
y
-
c
-
m
e
a
n
s
1.7.2
g
a
s
t
0.6.0
g
a
t
s
p
y
0.3
g
e
n
s
i
m
4.3.3
g
e
o
p
a
n
d
a
s
1.0.1
g
e
v
e
n
t
2
4.1
1.1
g
i
t
d
b
4.0.1
1
G
i
t
P
y
t
h
o
n
3.1.4
3
g
l
u
o
n
t
s
0.1
6.0
g
o
o
g
l
e
-
a
i
-
g
e
n
e
r
a
t
i
v
e
l
a
n
g
u
a
g
e
0.6.1
0
g
o
o
g
l
e
-
a
p
i
-
c
o
r
e
2.2
4.0
g
o
o
g
l
e
-
a
p
i
-
p
y
t
h
o
n
-
c
l
i
e
n
t
2.1
5
4.0
g
o
o
g
l
e
-
a
u
t
h
2.3
6.0
g
o
o
g
l
e
-
a
u
t
h
-
h
t
t
p
l
i
b
2
0.2.0
g
o
o
g
l
e
-
g
e
n
e
r
a
t
i
v
e
a
i
0.8.3
g
o
o
g
l
e
-
p
a
s
t
a
0.2.0
g
o
o
g
l
e
a
p
i
s
-
c
o
m
m
o
n
-
p
r
o
t
o
s
1.6
6.0
g
p
f
l
o
w
2.9.2
g
p
l
e
a
r
n
0.4.2
g
p
y
t
o
r
c
h
1.1
3
g
r
a
p
h
e
n
e
3.4.3
g
r
a
p
h
q
l
-
c
o
r
e
3.2.5
g
r
a
p
h
q
l
-
r
e
l
a
y
3.2.0
g
r
a
p
h
v
i
z
0.2
0.3
g
r
e
e
n
l
e
t
3.1.1
g
r
p
c
i
o
1.6
8.1
g
r
p
c
i
o
-
s
t
a
t
u
s
1.6
8.1
g
u
n
i
c
o
r
n
2
3.0.0
g
y
m
0.2
6.2
g
y
m
-
n
o
t
i
c
e
s
0.0.8
g
y
m
n
a
s
i
u
m
1.0.0
h
1
1
0.1
4.0
h
2
o
3.4
6.0.6
h
5
n
e
t
c
d
f
1.4.1
h
5
p
y
3.1
2.1
h
m
m
l
e
a
r
n
0.3.3
h
o
l
i
d
a
y
s
0.6
2
h
o
l
o
v
i
e
w
s
1.2
0.0
h
o
m
e
b
a
s
e
1.0.1
h
o
p
c
r
o
f
t
k
a
r
p
1.2.5
h
t
m
l
5
l
i
b
1.1
h
t
t
p
c
o
r
e
1.0.7
h
t
t
p
l
i
b
2
0.2
2.0
h
t
t
p
s
t
a
n
4.1
3.0
h
t
t
p
x
0.2
8.1
h
u
g
g
i
n
g
f
a
c
e
-
h
u
b
0.2
6.5
h
u
r
s
t
0.0.5
h
v
p
l
o
t
0.1
1.1
h
y
p
e
r
o
p
t
0.2.7
i
b
m
-
c
l
o
u
d
-
s
d
k
-
c
o
r
e
3.2
2.0
i
b
m
-
p
l
a
t
f
o
r
m
-
s
e
r
v
i
c
e
s
0.5
9.0
i
d
n
a
3.7
i
i
s
i
g
n
a
t
u
r
e
0.2
4
i
j
s
o
n
3.3.0
i
m
a
g
e
i
o
2.3
6.1
i
m
b
a
l
a
n
c
e
d
-
l
e
a
r
n
0.1
2.4
i
m
m
u
t
a
b
l
e
d
i
c
t
4.2.1
i
m
p
o
r
t
l
i
b
_
m
e
t
a
d
a
t
a
8.5.0
i
m
p
o
r
t
l
i
b
_
r
e
s
o
u
r
c
e
s
6.4.5
i
n
i
c
o
n
f
i
g
2.0.0
i
n
j
e
c
t
o
r
0.2
2.0
i
n
t
e
r
f
a
c
e
-
m
e
t
a
1.3.0
i
n
t
e
r
p
r
e
t
0.6.7
i
n
t
e
r
p
r
e
t
-
c
o
r
e
0.6.7
i
p
y
k
e
r
n
e
l
6.2
9.5
i
p
y
m
p
l
0.9.4
i
p
y
t
h
o
n
8.3
0.0
i
p
y
t
h
o
n
-
g
e
n
u
t
i
l
s
0.2.0
i
p
y
w
i
d
g
e
t
s
8.1.5
i
s
o
d
u
r
a
t
i
o
n
2
0.1
1.0
i
t
s
d
a
n
g
e
r
o
u
s
2.2.0
j
a
x
0.4.3
5
j
a
x
l
i
b
0.4.3
5
j
a
x
t
y
p
i
n
g
0.2.1
9
j
e
d
i
0.1
9.2
J
i
n
j
a
2
3.1.4
j
i
t
e
r
0.8.2
j
o
b
l
i
b
1.3.2
j
s
o
n
5
0.1
0.0
json5 0.10.0
jsonpatch 1.33
jsonpath-ng 1.7.0
jsonpointer 2.1
jsonschema 4.23.0
jsonschema-specifications 2024.10.1
jupyter 1.1.1
jupyter_ai 2.28.2
jupyter_ai_magics 2.28.3
jupyter_bokeh 4.0.5
jupyter_client 8.6.3
jupyter-console 6.6.3
jupyter_core 5.7.2
jupyter-events 0.10.0
jupyter-lsp 2.2.5
jupyter-resource-usage 1.1.0
jupyter_server 2.14.2
jupyter_server_proxy 4.4.0
jupyter_server_terminals 0.5.3
jupyterlab 4.3.2
jupyterlab_pygments 0.3.0
jupyterlab_server 2.27.3
jupyterlab_widgets 3.0.13
kagglehub 0.3.4
kaleido 0.2.1
keras 3.7.0
keras-hub 0.18.1
keras-nlp 0.18.1
keras-rl 0.4.2
keras-tcn 3.5.0
keras-tuner 1.4.7
kiwisolver 1.4.7
kmapper 2.1.0
korean-lunar-calendar 0.3.1
kt-legacy 1.0.5
langchain 0.2.17
langchain-community 0.2.19
langchain-core 0.2.43
langchain-text-splitters 0.2.4
langcodes 3.5.0
langsmith 0.1.147
language_data 1.3.0
lark 1.2.2
lazy_loader 0.4
lazypredict 0.2.14a1
libclang 18.1.1
libmambapy 1.5.8
libucx-cu12 1.15.0.post2
lifelines 0.30.0
lightgbm 4.5.0
lightning 2.4.0
lightning-utilities 0.11.9
lime 0.2.0.1
line_profiler 4.2.0
linear-operator 0.5.3
linkify-it-py 2.0.3
livelossplot 0.5.5
llama-cloud 0.1.6
llama-index 0.12.2
llama-index-agent-openai 0.4.0
llama-index-cli 0.4.0
llama-index-core 0.12.5
llama-index-embeddings-openai 0.3.1
llama-index-indices-managed-llama-cloud 0.6.3
llama-index-legacy 0.9.48.post4
llama-index-llms-openai 0.3.3
llama-index-multi-modal-llms-openai 0.3.0
llama-index-program-openai 0.3.1
llama-index-question-gen-openai 0.3.0
llama-index-readers-file 0.4.1
llama-index-readers-llama-parse 0.4.0
llama-parse 0.5.17
llvmlite 0.42.0
locket 1.0.0
logical-unification 0.4.6
loguru 0.7.3
lxml 5.3.0
lz4 4.3.3
Mako 1.3.8
mamba-ssm 2.2.4
MAPIE 0.9.1
marisa-trie 1.2.1
marisa-trie 1.2.1
Markdown 3.7
markdown-it-py 3.0.0
MarkupSafe 3.0.2
marshmallow 3.23.1
matplotlib 3.7.5
matplotlib-inline 0.1.7
mdit-py-plugins 0.4.2
mdurl 0.1.2
menuinst 2.1.2
mgarch 0.3.0
miniKanren 1.0.3
minorminer 0.2.15
mistune 3.0.2
ml-dtypes 0.4.1
mlflow 2.18.0
mlflow-skinny 2.18.0
mlforecast 0.15.1
mljar-scikit-plot 0.3.12
mljar-supervised 1.1.9
mlxtend 0.23.3
mmh3 2.5.1
modin 0.26.1
mplfinance 0.12.10b0
mpmath 1.3.0
msgpack 1.1.0
multidict 6.1.0
multipledispatch 1.0.0
multiprocess 0.70.16
multitasking 0.0.11
murmurhash 1.0.11
mypy-extensions 1.0.0
namex 0.0.8
narwhals 1.17.0
nbclient 0.10.1
nbconvert 7.16.4
nbformat 5.10.4
ndindex 1.9.2
nest-asyncio 1.6.0
networkx 3.4.2
neural-tangents 0.6.5
neuralprophet 0.9.0
nfoursid 1.0.1
ngboost 0.5.1
ninja 1.11.1.2
nixtla 0.6.4
nltk 3.9.1
nolds 0.6.1
nose 1.3.7
notebook 7.3.1
notebook_shim 0.2.4
numba 0.59.1
numerapi 2.19.1
numexpr 2.10.2
numpy 1.26.4
nvidia-cublas-cu12 12.4.5.8
nvidia-cuda-cupti-cu12 12.4.127
nvidia-cuda-nvrtc-cu12 12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12 9.3.0.75
nvidia-cufft-cu12 11.2.1.3
nvidia-curand-cu12 10.3.5.147
nvidia-cusolver-cu12 11.6.1.9
nvidia-cusparse-cu12 12.3.1.170
nvidia-nccl-cu12 2.21.5
nvidia-nvjitlink-cu12 12.4.127
nvidia-nvtx-cu12 12.4.127
nvtx 0.2.10
nx-cugraph-cu12 24.8.0
oauthlib 3.2.2
openai 1.57.0
opencv-contrib-python-headless 4.10.0.84
opencv-python 4.10.0.84
openpyxl 3.1.5
opentelemetry-api 1.28.2
opentelemetry-sdk 1.28.2
opentelemetry-semantic-conventions 0.49b2
opt_einsum 3.4.0
optree 0.13.1
optuna 4.1.0
orjson 3.10.12
ortools 9.9.3963
o
r
t
o
o
l
s
9.9.3
9
6
3
o
s
q
p
0.6.7.p
o
s
t
3
o
v
e
r
r
i
d
e
s
7.7.0
p
a
c
k
a
g
i
n
g
2
4.1
p
a
n
d
a
s
2.1.4
p
a
n
d
a
s
-
f
l
a
v
o
r
0.6.0
p
a
n
d
a
s
_
m
a
r
k
e
t
_
c
a
l
e
n
d
a
r
s
4.4.2
p
a
n
d
a
s
_
t
a
0.3.1
4
b
0
p
a
n
d
o
c
f
i
l
t
e
r
s
1.5.1
p
a
n
e
l
1.5.4
p
a
r
a
m
2.1.1
p
a
r
s
o
0.8.4
p
a
r
t
d
1.4.2
p
a
s
t
e
l
0.2.1
p
a
t
h
o
s
0.3.2
p
a
t
s
y
1.0.1
p
b
r
6.1.0
p
e
e
w
e
e
3.1
7.3
p
e
f
t
0.1
3.2
p
e
n
a
l
t
y
m
o
d
e
l
1.1.0
P
e
n
n
y
L
a
n
e
0.3
9.0
P
e
n
n
y
L
a
n
e
_
L
i
g
h
t
n
i
n
g
0.3
9.0
P
e
n
n
y
L
a
n
e
-
q
i
s
k
i
t
0.3
6.0
p
e
r
s
i
m
0.3.7
p
e
x
p
e
c
t
4.9.0
p
g
m
p
y
0.1.2
6
p
i
l
l
o
w
1
0.4.0
p
i
n
g
o
u
i
n
0.5.5
p
i
p
2
4.2
p
l
a
t
f
o
r
m
d
i
r
s
3.1
0.0
p
l
o
t
l
y
5.2
4.1
p
l
o
t
l
y
-
r
e
s
a
m
p
l
e
r
0.1
0.0
p
l
u
c
k
y
0.4.3
p
l
u
g
g
y
1.5.0
p
l
y
3.1
1
p
m
d
a
r
i
m
a
2.0.4
p
o
l
a
r
s
1.1
6.0
p
o
m
e
g
r
a
n
a
t
e
1.1.1
P
O
T
0.9.5
p
o
x
0.3.5
p
p
f
t
1.7.6.9
p
p
r
o
f
i
l
e
2.2.0
p
r
e
s
h
e
d
3.0.9
p
r
o
m
e
t
h
e
u
s
_
c
l
i
e
n
t
0.2
1.1
p
r
o
m
p
t
_
t
o
o
l
k
i
t
3.0.4
8
p
r
o
p
c
a
c
h
e
0.2.1
p
r
o
p
h
e
t
1.1.6
p
r
o
t
o
-
p
l
u
s
1.2
5.0
p
r
o
t
o
b
u
f
5.2
9.1
p
s
u
t
i
l
5.9.8
p
t
y
p
r
o
c
e
s
s
0.7.0
P
u
L
P
2.9.0
p
u
r
e
_
e
v
a
l
0.2.3
p
y
-
c
p
u
i
n
f
o
9.0.0
p
y
-
h
e
a
t
0.0.6
p
y
-
h
e
a
t
-
m
a
g
i
c
0.0.2
p
y
_
l
e
t
s
_
b
e
_
r
a
t
i
o
n
a
l
1.0.1
p
y
_
v
o
l
l
i
b
1.0.1
p
y
4
j
0.1
0.9.7
p
y
a
m
l
2
4.9.0
p
y
a
r
r
o
w
1
6.1.0
p
y
a
s
n
1
0.6.1
p
y
a
s
n
1
_
m
o
d
u
l
e
s
0.4.1
p
y
b
i
n
d
1
1
2.1
3.6
p
y
c
a
r
e
t
3.3.2
p
y
c
o
s
a
t
0.6.6
p
y
c
p
a
r
s
e
r
2.2
1
p
y
c
t
0.5.0
p
y
d
a
n
t
i
c
2.9.2
p
y
d
a
n
t
i
c
_
c
o
r
e
2.2
3.4
P
y
D
M
D
2
0
2
4.1
2.1
p
y
e
r
f
a
2.0.1.5
p
y
f
o
l
i
o
-
r
e
l
o
a
d
e
d
0.9.8
P
y
g
m
e
n
t
s
2.1
8.0
P
y
J
W
T
2.1
0.1
p
y
k
a
l
m
a
n
0.9.7
p
y
l
e
v
1.4.0
p
y
l
i
b
c
u
g
r
a
p
h
-
c
u
1
2
2
4.8.0
p
y
l
i
b
r
a
f
t
-
c
u
1
2
2
4.8.1
p
y
l
u
a
c
h
2.2.0
p
y
m
a
n
n
k
e
n
d
a
l
l
1.4.3
p
y
m
c
5.1
9.0
p
y
m
c
5.1
9.0
p
y
m
d
p
t
o
o
l
b
o
x
4.0
b
3
p
y
n
n
d
e
s
c
e
n
t
0.5.1
3
p
y
n
v
j
i
t
l
i
n
k
-
c
u
1
2
0.4.0
p
y
n
v
m
l
1
1.4.1
p
y
o
d
2.0.2
p
y
o
g
r
i
o
0.1
0.0
P
y
o
m
o
6.8.2
p
y
p
a
r
s
i
n
g
3.2.0
p
y
p
d
f
5.1.0
p
y
p
o
r
t
f
o
l
i
o
o
p
t
1.5.6
p
y
p
r
o
j
3.7.0
P
y
Q
t
6
6.7.1
P
y
Q
t
6
-
Q
t
6
6.7.3
P
y
Q
t
6
_
s
i
p
1
3.9.0
p
y
r
b
1.0.1
p
y
r
e
-
e
x
t
e
n
s
i
o
n
s
0.0.3
2
p
y
r
o
-
a
p
i
0.1.2
p
y
r
o
-
p
p
l
1.9.1
p
y
s
i
m
d
j
s
o
n
6.0.2
P
y
S
o
c
k
s
1.7.1
p
y
s
p
n
e
g
o
0.1
1.2
p
y
s
t
a
n
3.1
0.0
p
y
t
e
n
s
o
r
2.2
6.4
p
y
t
e
s
t
8.3.4
p
y
t
e
s
t
-
r
u
n
n
e
r
6.0.1
p
y
t
h
o
n
-
d
a
t
e
u
t
i
l
2.9.0.p
o
s
t
0
p
y
t
h
o
n
-
j
s
o
n
-
l
o
g
g
e
r
2.0.7
p
y
t
h
o
n
-
s
t
a
t
e
m
a
c
h
i
n
e
2.5.0
p
y
t
o
r
c
h
-
f
o
r
e
c
a
s
t
i
n
g
1.2.0
p
y
t
o
r
c
h
-
i
g
n
i
t
e
0.5.1
p
y
t
o
r
c
h
-
l
i
g
h
t
n
i
n
g
2.4.0
p
y
t
o
r
c
h
-
t
a
b
n
e
t
4.1.0
p
y
t
z
2
0
2
4.2
p
y
v
i
n
e
c
o
p
u
l
i
b
0.6.5
p
y
v
i
z
_
c
o
m
m
s
3.0.3
P
y
W
a
v
e
l
e
t
s
1.7.0
P
y
Y
A
M
L
6.0.2
p
y
z
m
q
2
6.2.0
q
d
l
d
l
0.1.7.p
o
s
t
4
q
i
s
k
i
t
1.2.4
q
i
s
k
i
t
-
a
e
r
0.1
5.1
q
i
s
k
i
t
-
i
b
m
-
p
r
o
v
i
d
e
r
0.1
1.0
q
i
s
k
i
t
-
i
b
m
-
r
u
n
t
i
m
e
0.3
4.0
q
u
a
d
p
r
o
g
0.1.1
3
q
u
a
n
t
e
c
o
n
0.7.2
Q
u
a
n
t
L
i
b
1.3
6
Q
u
a
n
t
S
t
a
t
s
0.0.6
4
r
a
f
t
-
d
a
s
k
-
c
u
1
2
2
4.8.1
r
a
p
i
d
s
-
d
a
s
k
-
d
e
p
e
n
d
e
n
c
y
2
4.8.0
r
a
u
t
h
0.7.3
r
a
y
2.4
0.0
R
b
e
a
s
t
0.1.2
3
r
e
f
e
r
e
n
c
i
n
g
0.3
5.1
r
e
g
e
x
2
0
2
4.1
1.6
r
e
q
u
e
s
t
s
2.3
2.3
r
e
q
u
e
s
t
s
_
n
t
l
m
1.3.0
r
e
q
u
e
s
t
s
-
o
a
u
t
h
l
i
b
1.3.1
r
e
q
u
e
s
t
s
-
t
o
o
l
b
e
l
t
1.0.0
r
f
c
3
3
3
9
-
v
a
l
i
d
a
t
o
r
0.1.4
r
f
c
3
9
8
6
-
v
a
l
i
d
a
t
o
r
0.1.1
r
i
c
h
1
3.9.4
r
i
p
s
e
r
0.6.1
0
R
i
s
k
f
o
l
i
o
-
L
i
b
6.1.1
r
i
s
k
p
a
r
i
t
y
p
o
r
t
f
o
l
i
o
0.6.0
r
i
v
e
r
0.2
1.0
r
m
m
-
c
u
1
2
2
4.8.2
r
p
d
s
-
p
y
0.2
2.3
r
s
a
4.9
r
u
a
m
e
l.y
a
m
l
0.1
8.6
r
u
a
m
e
l.y
a
m
l.c
l
i
b
0.2.8
r
u
p
t
u
r
e
s
1.1.9
r
u
s
t
w
o
r
k
x
0.1
5.1
s
a
f
e
t
e
n
s
o
r
s
0.4.5
S
A
L
i
b
1.5.1
s
c
h
e
m
d
r
a
w
0.1
5
s
c
i
k
e
r
a
s
0.1
3.0
s
c
i
k
i
t
-
b
a
s
e
0.7.8
s
c
i
k
i
t
-
i
m
a
g
e
0.2
2.0
s
c
i
k
i
t
-
l
e
a
r
n
1.4.2
s
c
i
k
i
t
-
l
e
a
r
n
-
e
x
t
r
a
0.3.0
s
c
i
k
i
t
-
o
p
t
i
m
i
z
e
0.1
0.2
s
c
i
k
i
t
-
o
p
t
i
m
i
z
e
0.1
0.2
s
c
i
k
i
t
-
p
l
o
t
0.3.7
s
c
i
k
i
t
-
t
d
a
1.1.1
s
c
i
p
y
1.1
1.4
s
c
s
3.2.7
s
d
e
i
n
t
0.3.0
s
e
a
b
o
r
n
0.1
3.2
S
e
n
d
2
T
r
a
s
h
1.8.3
s
e
n
t
e
n
c
e
-
t
r
a
n
s
f
o
r
m
e
r
s
3.3.1
s
e
t
u
p
t
o
o
l
s
7
3.0.1
s
e
t
u
p
t
o
o
l
s
-
s
c
m
8.1.0
s
h
a
p
0.4
6.0
s
h
a
p
e
l
y
2.0.6
S
h
i
m
m
y
2.0.0
s
i
m
p
e
r
v
i
s
o
r
1.0.0
s
i
m
p
l
e
j
s
o
n
3.1
9.3
s
i
m
p
y
4.1.1
s
i
x
1.1
7.0
s
k
l
e
a
r
n
-
j
s
o
n
0.1.0
s
k
t
i
m
e
0.2
6.0
s
l
i
c
e
r
0.0.8
s
m
a
r
t
-
o
p
e
n
7.0.5
s
m
m
a
p
5.0.1
s
n
i
f
f
i
o
1.3.1
s
o
r
t
e
d
c
o
n
t
a
i
n
e
r
s
2.4.0
s
o
u
p
s
i
e
v
e
2.6
s
p
a
c
y
3.7.5
s
p
a
c
y
-
l
e
g
a
c
y
3.0.1
2
s
p
a
c
y
-
l
o
g
g
e
r
s
1.0.5
S
Q
L
A
l
c
h
e
m
y
2.0.3
6
s
q
l
p
a
r
s
e
0.5.3
s
r
s
l
y
2.5.0
s
s
m
0.0.1
s
t
a
b
l
e
_
b
a
s
e
l
i
n
e
s
3
2.4.0
s
t
a
c
k
-
d
a
t
a
0.6.3
s
t
a
n
i
o
0.5.1
s
t
a
t
s
f
o
r
e
c
a
s
t
2.0.0
s
t
a
t
s
m
o
d
e
l
s
0.1
4.4
s
t
e
v
e
d
o
r
e
5.4.0
s
t
o
c
h
a
s
t
i
c
0.6.0
s
t
o
c
k
s
t
a
t
s
0.6.2
s
t
o
p
i
t
1.1.2
s
t
r
i
p
r
t
f
0.0.2
6
s
t
u
m
p
y
1.1
3.0
s
y
m
e
n
g
i
n
e
0.1
3.0
s
y
m
p
y
1.1
3.1
t
a
0.1
1.0
t
a
-
l
i
b
0.5.1
t
a
b
l
e
s
3.1
0.1
t
a
b
u
l
a
t
e
0.8.1
0
t
a
d
a
s
e
t
s
0.2.1
t
b
a
t
s
1.1.3
t
b
l
i
b
3.0.0
t
e
n
a
c
i
t
y
8.5.0
t
e
n
s
o
r
b
o
a
r
d
2.1
8.0
t
e
n
s
o
r
b
o
a
r
d
-
d
a
t
a
-
s
e
r
v
e
r
0.7.2
t
e
n
s
o
r
b
o
a
r
d
X
2.6.2.2
t
e
n
s
o
r
f
l
o
w
2.1
8.0
t
e
n
s
o
r
f
l
o
w
-
a
d
d
o
n
s
0.2
3.0
t
e
n
s
o
r
f
l
o
w
_
d
e
c
i
s
i
o
n
_
f
o
r
e
s
t
s
1.1
1.0
t
e
n
s
o
r
f
l
o
w
-
i
o
-
g
c
s
-
f
i
l
e
s
y
s
t
e
m
0.3
7.1
t
e
n
s
o
r
f
l
o
w
-
p
r
o
b
a
b
i
l
i
t
y
0.2
5.0
t
e
n
s
o
r
f
l
o
w
-
t
e
x
t
2.1
8.0
t
e
n
s
o
r
l
y
0.9.0
t
e
n
s
o
r
r
t
1
0.7.0
t
e
n
s
o
r
r
t
_
c
u
1
2
1
0.7.0
t
e
n
s
o
r
r
t
-
c
u
1
2
-
b
i
n
d
i
n
g
s
1
0.7.0
t
e
n
s
o
r
r
t
-
c
u
1
2
-
l
i
b
s
1
0.7.0
t
e
n
s
o
r
t
r
a
d
e
1.0.3
t
e
r
m
c
o
l
o
r
2.5.0
t
e
r
m
i
n
a
d
o
0.1
8.1
t
f
_
k
e
r
a
s
2.1
8.0
t
f
2
j
a
x
0.3.6
t
h
i
n
c
8.2.5
t
h
r
e
a
d
p
o
o
l
c
t
l
3.5.0
t
h
u
n
d
e
r
g
b
m
0.3.1
7
t
i
f
f
f
i
l
e
2
0
2
4.9.2
0
t
i
g
r
a
m
i
t
e
5.2.6.7
t
i
k
t
o
k
e
n
0.8.0
t
i
n
y
c
s
s
2
1.4.0
t
i
n
y
g
r
a
d
0.1
0.0
t
o
k
e
n
i
z
e
r
s
0.2
0.3
Legacy Environment
tokenizers 0.20.3
toml 0.10.2
toolz 0.12.1
torch 2.5.1
torch_cluster 1.6.3
torch-geometric 2.6.1
torch_scatter 2.1.2
torch_sparse 0.6.18
torch_spline_conv 1.2.2
torchdata 0.10.0
torchmetrics 1.6.0
torchvision 0.20.1
tornado 6.4.2
TPOT 0.12.2
tqdm 4.66.5
traitlets 5.14.3
transformers 4.46.3
treelite 4.3.0
triad 0.9.8
triton 3.1.0
truststore 0.8.0
tsdownsample 0.1.3
tsfel 0.1.9
tsfresh 0.20.2
tslearn 0.6.3
tweepy 4.14.0
typeguard 2.13.3
typer 0.9.4
typer-config 1.4.2
types-python-dateutil 2.9.0.20241206
typing_extensions 4.12.2
typing-inspect 0.9.0
tzdata 2024.2
uc-micro-py 1.0.3
ucx-py-cu12 0.39.2
ucxx-cu12 0.39.1
umap-learn 0.5.7
update-checker 0.18.0
uri-template 1.3.0
uritemplate 4.1.1
urllib3 2.2.3
utilsforecast 0.2.10
wasabi 1.1.3
wcwidth 0.2.13
weasel 0.4.1
webargs 8.6.0
webcolors 24.11.1
webencodings 0.5.1
websocket-client 1.8.0
websockets 14.1
Werkzeug 3.1.3
wheel 0.44.0
widgetsnbextension 4.0.13
window_ops 0.0.15
woodwork 0.31.0
wordcloud 1.9.4
wrapt 1.16.0
wurlitzer 3.1.1
x-transformers 1.42.24
xarray 2024.11.0
xarray-einstats 0.8.0
xgboost 2.1.3
xlrd 2.0.1
XlsxWriter 3.2.0
xxhash 3.5.0
xyzservices 2024.9.0
yarl 1.18.3
ydf 0.9.0
yellowbrick 1.5
yfinance 0.2.50
zict 3.0.0
zipp 3.21.0
zope.event 5.0
zope.interface 7.2
zstandard 0.23.0
The Legacy environment provides a backwards compatability environment for compatibility with Keras < 3 ,
Tensorflow 2.14.1 , and Pydantic . This environment supports the following libraries:
absl-py 2.1.0
accelerate 0.34.2
adagio 0.2.6
aesara 2.9.4
aiohappyeyeballs 2.4.4
aiohttp 3.11.10
aiosignal 1.3.1
aiosqlite 0.20.0
alembic 1.14.0
alibi-detect 0.12.0
alphalens-reloaded 0.4.5
altair 5.5.0
anaconda-anon-usage 0.4.4
annotated-types 0.7.0
antlr4-python3-runtime 4.9.3
anyio 4.7.0
aplr 10.8.0
appdirs 1.4.4
apricot-select 0.6.1
arch 7.2.0
archspec 0.2.3
argon2-cffi 23.1.0
argon2-cffi-bindings 21.2.0
arrow 1.3.0
arviz 0.20.0
astropy 7.0.0
astropy-iers-data 0.2024.12.9.0.36.21
asttokens 3.0.0
astunparse 1.6.3
async-lru 2.0.4
attrs 24.2.0
Authlib 1.3.2
autograd 1.7.0
autograd-gamma 0.5.0
autokeras 2.0.0
autoray 0.7.0
ax-platform 0.4.3
babel 2.16.0
bayesian-optimization 2.0.0
beautifulsoup4 4.12.3
bleach 6.2.0
blinker 1.9.0
blis 0.7.11
blosc2 2.7.1
bokeh 3.6.2
boltons 23.0.0
botorch 0.12.0
Bottleneck 1.4.2
Brotli 1.0.9
cachetools 5.5.0
captum 0.7.0
catalogue 2.0.10
catboost 1.2.7
category-encoders 2.6.4
causal-conv1d 1.5.0.post8
certifi 2024.8.30
cesium 0.12.1
cffi 1.17.1
chardet 5.2.0
charset-normalizer 3.3.2
check-shapes 1.1.1
chronos-forecasting 1.4.1
clarabel 0.9.0
click 8.1.7
clikit 0.6.2
cloudpathlib 0.20.0
cloudpickle 3.1.0
cmdstanpy 1.2.4
colorama 0.4.6
colorcet 3.1.0
colorlog 6.9.0
colorlover 0.3.0
colour 0.1.5
comm 0.2.2
conda 24.9.2
PY
c
o
n
d
a
2
4.9.2
c
o
n
d
a
-
c
o
n
t
e
n
t
-
t
r
u
s
t
0.2.0
c
o
n
d
a
-
l
i
b
m
a
m
b
a
-
s
o
l
v
e
r
2
4.9.0
c
o
n
d
a
-
p
a
c
k
a
g
e
-
h
a
n
d
l
i
n
g
2.3.0
c
o
n
d
a
_
p
a
c
k
a
g
e
_
s
t
r
e
a
m
i
n
g
0.1
0.0
c
o
n
f
e
c
t
i
o
n
0.1.5
c
o
n
s
0.4.6
c
o
n
t
o
u
r
p
y
1.3.1
c
o
n
t
r
o
l
0.1
0.1
c
o
p
u
l
a
e
0.7.9
c
o
p
u
l
a
s
0.1
2.0
c
o
r
e
f
o
r
e
c
a
s
t
0.0.1
5
c
r
a
m
j
a
m
2.9.0
c
r
a
s
h
t
e
s
t
0.3.1
c
r
e
m
e
0.6.1
c
r
y
p
t
o
g
r
a
p
h
y
4
3.0.0
c
u
c
i
m
-
c
u
1
2
2
4.8.0
c
u
d
a
-
p
y
t
h
o
n
1
2.6.2.p
o
s
t
1
c
u
d
f
-
c
u
1
2
2
4.8.3
c
u
f
f
l
i
n
k
s
0.1
7.3
c
u
g
r
a
p
h
-
c
u
1
2
2
4.8.0
c
u
m
l
-
c
u
1
2
2
4.8.0
c
u
p
r
o
j
-
c
u
1
2
2
4.8.0
c
u
p
y
-
c
u
d
a
1
2
x
1
3.3.0
c
u
s
p
a
t
i
a
l
-
c
u
1
2
2
4.8.0
c
u
v
s
-
c
u
1
2
2
4.8.0
c
u
x
f
i
l
t
e
r
-
c
u
1
2
2
4.8.0
c
v
x
o
p
t
1.3.2
c
v
x
p
o
r
t
f
o
l
i
o
1.4.0
c
v
x
p
y
1.6.0
c
y
c
l
e
r
0.1
2.1
c
y
m
e
m
2.0.1
0
C
y
t
h
o
n
3.0.1
1
d
a
r
t
s
0.3
1.0
d
a
s
h
2.9.3
d
a
s
h
-
c
o
r
e
-
c
o
m
p
o
n
e
n
t
s
2.0.0
d
a
s
h
_
c
y
t
o
s
c
a
p
e
1.0.2
d
a
s
h
-
h
t
m
l
-
c
o
m
p
o
n
e
n
t
s
2.0.0
d
a
s
h
-
t
a
b
l
e
5.0.0
d
a
s
k
2
0
2
4.7.1
d
a
s
k
-
c
u
d
a
2
4.8.2
d
a
s
k
-
c
u
d
f
-
c
u
1
2
2
4.8.3
d
a
s
k
-
e
x
p
r
1.1.9
d
a
t
a
b
r
i
c
k
s
-
s
d
k
0.3
8.0
d
a
t
a
c
l
a
s
s
e
s
-
j
s
o
n
0.6.7
d
a
t
a
s
e
t
s
2.1
7.1
d
a
t
a
s
h
a
d
e
r
0.1
6.3
d
e
a
p
1.4.1
d
e
b
u
g
p
y
1.8.9
d
e
c
o
r
a
t
o
r
5.1.1
d
e
e
p
m
e
r
g
e
2.0
d
e
f
u
s
e
d
x
m
l
0.7.1
D
e
p
r
e
c
a
t
e
d
1.2.1
5
d
e
p
r
e
c
a
t
i
o
n
2.1.0
d
g
l
2.1.0
d
i
l
l
0.3.8
d
i
m
o
d
0.1
2.1
7
d
i
r
t
y
j
s
o
n
1.0.8
d
i
s
k
c
a
c
h
e
5.6.3
d
i
s
t
r
i
b
u
t
e
d
2
0
2
4.7.1
d
i
s
t
r
i
b
u
t
e
d
-
u
c
x
x
-
c
u
1
2
0.3
9.1
d
i
s
t
r
o
1.9.0
d
m
-
t
r
e
e
0.1.8
d
o
c
k
e
r
7.1.0
d
o
c
u
t
i
l
s
0.2
1.2
D
o
u
b
l
e
M
L
0.9.0
d
r
o
p
s
t
a
c
k
f
r
a
m
e
0.1.1
d
t
r
e
e
v
i
z
2.2.2
d
t
w
-
p
y
t
h
o
n
1.5.3
d
w
a
v
e
-
c
l
o
u
d
-
c
l
i
e
n
t
0.1
3.1
d
w
a
v
e
-
d
r
i
v
e
r
s
0.4.4
d
w
a
v
e
-
g
a
t
e
0.3.2
d
w
a
v
e
-
g
r
e
e
d
y
0.3.0
d
w
a
v
e
-
h
y
b
r
i
d
0.6.1
2
d
w
a
v
e
-
i
n
s
p
e
c
t
o
r
0.5.1
d
w
a
v
e
-
i
n
s
p
e
c
t
o
r
a
p
p
0.3.3
d
w
a
v
e
-
n
e
a
l
0.6.0
d
w
a
v
e
_
n
e
t
w
o
r
k
x
0.8.1
5
d
w
a
v
e
-
o
c
e
a
n
-
s
d
k
8.0.1
d
w
a
v
e
-
o
p
t
i
m
i
z
a
t
i
o
n
0.3.0
d
w
a
v
e
-
p
r
e
p
r
o
c
e
s
s
i
n
g
0.6.6
d
w
a
v
e
-
s
a
m
p
l
e
r
s
1.3.0
d
w
a
v
e
-
s
a
m
p
l
e
r
s
1.3.0
d
w
a
v
e
-
s
y
s
t
e
m
1.2
6.0
d
w
a
v
e
-
t
a
b
u
0.5.0
d
w
a
v
e
b
i
n
a
r
y
c
s
p
0.3.0
e
c
o
s
2.0.1
4
e
i
n
o
p
s
0.7.0
e
i
n
x
0.3.0
E
M
D
-
s
i
g
n
a
l
1.6.4
e
m
p
y
r
i
c
a
l
-
r
e
l
o
a
d
e
d
0.5.1
1
e
n
-
c
o
r
e
-
w
e
b
-
m
d
3.7.1
e
n
-
c
o
r
e
-
w
e
b
-
s
m
3.7.1
e
t
_
x
m
l
f
i
l
e
2.0.0
e
t
u
p
l
e
s
0.3.9
e
x
c
h
a
n
g
e
_
c
a
l
e
n
d
a
r
s
4.6
e
x
e
c
u
t
i
n
g
2.1.0
f
a
i
s
s
-
c
p
u
1.9.0.p
o
s
t
1
F
a
r
a
m
a
-
N
o
t
i
f
i
c
a
t
i
o
n
s
0.0.4
f
a
s
t
a
i
2.7.1
8
f
a
s
t
a
i
2
0.0.3
0
f
a
s
t
c
o
r
e
1.7.2
6
f
a
s
t
d
o
w
n
l
o
a
d
0.0.7
f
a
s
t
e
n
e
r
s
0.1
9
f
a
s
t
j
s
o
n
s
c
h
e
m
a
2.2
1.1
f
a
s
t
p
a
r
q
u
e
t
2
0
2
4.1
1.0
f
a
s
t
p
r
o
g
r
e
s
s
1.0.3
f
a
s
t
r
l
o
c
k
0.8.2
f
a
s
t
t
e
x
t
0.9.3
f
e
a
t
u
r
e
-
e
n
g
i
n
e
1.6.2
f
e
a
t
u
r
e
t
o
o
l
s
1.3
1.0
f
i
l
e
l
o
c
k
3.1
6.1
f
i
l
e
t
y
p
e
1.2.0
f
i
n
d
i
f
f
0.1
0.2
F
i
x
e
d
E
f
f
e
c
t
M
o
d
e
l
0.0.5
F
l
a
g
E
m
b
e
d
d
i
n
g
1.2.1
1
F
L
A
M
L
2.3.3
F
l
a
s
k
3.1.0
f
l
a
t
b
u
f
f
e
r
s
2
4.3.2
5
f
o
n
t
t
o
o
l
s
4.5
5.2
f
o
r
m
u
l
a
i
c
1.0.2
f
q
d
n
1.5.1
f
r
o
z
e
n
d
i
c
t
2.4.2
f
r
o
z
e
n
l
i
s
t
1.5.0
f
s
2.4.1
6
f
s
s
p
e
c
2
0
2
3.1
0.0
f
u
g
u
e
0.9.1
f
u
n
c
t
i
m
e
0.9.5
f
u
t
u
r
e
1.0.0
f
u
z
z
y
-
c
-
m
e
a
n
s
1.7.2
g
a
s
t
0.6.0
g
a
t
s
p
y
0.3
g
e
n
s
i
m
4.3.3
g
e
o
p
a
n
d
a
s
1.0.1
g
e
v
e
n
t
2
4.1
1.1
g
i
t
d
b
4.0.1
1
G
i
t
P
y
t
h
o
n
3.1.4
3
g
l
u
o
n
t
s
0.1
4.4
g
o
o
g
l
e
-
a
i
-
g
e
n
e
r
a
t
i
v
e
l
a
n
g
u
a
g
e
0.6.1
0
g
o
o
g
l
e
-
a
p
i
-
c
o
r
e
2.2
4.0
g
o
o
g
l
e
-
a
p
i
-
p
y
t
h
o
n
-
c
l
i
e
n
t
2.1
5
4.0
g
o
o
g
l
e
-
a
u
t
h
2.3
6.0
g
o
o
g
l
e
-
a
u
t
h
-
h
t
t
p
l
i
b
2
0.2.0
g
o
o
g
l
e
-
a
u
t
h
-
o
a
u
t
h
l
i
b
1.0.0
g
o
o
g
l
e
-
g
e
n
e
r
a
t
i
v
e
a
i
0.8.3
g
o
o
g
l
e
-
p
a
s
t
a
0.2.0
g
o
o
g
l
e
a
p
i
s
-
c
o
m
m
o
n
-
p
r
o
t
o
s
1.6
6.0
g
p
f
l
o
w
2.9.2
g
p
l
e
a
r
n
0.4.2
g
p
y
t
o
r
c
h
1.1
3
g
r
a
p
h
e
n
e
3.4.3
g
r
a
p
h
q
l
-
c
o
r
e
3.2.5
g
r
a
p
h
q
l
-
r
e
l
a
y
3.2.0
g
r
a
p
h
v
i
z
0.2
0.3
g
r
e
e
n
l
e
t
3.1.1
g
r
p
c
i
o
1.6
8.1
g
r
p
c
i
o
-
s
t
a
t
u
s
1.6
8.1
g
u
n
i
c
o
r
n
2
3.0.0
g
y
m
0.2
6.2
g
y
m
-
n
o
t
i
c
e
s
0.0.8
g
y
m
n
a
s
i
u
m
1.0.0
h
1
1
0.1
4.0
h
2
o
3.4
6.0.6
h
5
n
e
t
c
d
f
1.4.1
h
5
n
e
t
c
d
f
1.4.1
h
5
p
y
3.1
2.1
h
m
m
l
e
a
r
n
0.3.3
h
o
l
i
d
a
y
s
0.6
2
h
o
l
o
v
i
e
w
s
1.2
0.0
h
o
m
e
b
a
s
e
1.0.1
h
o
p
c
r
o
f
t
k
a
r
p
1.2.5
h
t
m
l
5
l
i
b
1.1
h
t
t
p
c
o
r
e
1.0.7
h
t
t
p
l
i
b
2
0.2
2.0
h
t
t
p
s
t
a
n
4.1
3.0
h
t
t
p
x
0.2
8.1
h
u
g
g
i
n
g
f
a
c
e
-
h
u
b
0.2
6.5
h
u
r
s
t
0.0.5
h
v
p
l
o
t
0.1
1.1
h
y
d
r
a
-
c
o
r
e
1.3.0
h
y
p
e
r
o
p
t
0.2.7
i
b
m
-
c
l
o
u
d
-
s
d
k
-
c
o
r
e
3.2
2.0
i
b
m
-
p
l
a
t
f
o
r
m
-
s
e
r
v
i
c
e
s
0.5
9.0
i
d
n
a
3.7
i
i
s
i
g
n
a
t
u
r
e
0.2
4
i
j
s
o
n
3.3.0
i
m
a
g
e
i
o
2.3
6.1
i
m
b
a
l
a
n
c
e
d
-
l
e
a
r
n
0.1
2.4
i
m
m
u
t
a
b
l
e
d
i
c
t
4.2.1
i
m
p
o
r
t
l
i
b
-
m
e
t
a
d
a
t
a
4.1
3.0
i
m
p
o
r
t
l
i
b
_
r
e
s
o
u
r
c
e
s
6.4.5
i
n
i
c
o
n
f
i
g
2.0.0
i
n
j
e
c
t
o
r
0.2
2.0
i
n
t
e
r
f
a
c
e
-
m
e
t
a
1.3.0
i
n
t
e
r
p
r
e
t
0.6.7
i
n
t
e
r
p
r
e
t
-
c
o
r
e
0.6.7
i
p
y
k
e
r
n
e
l
6.2
9.5
i
p
y
m
p
l
0.9.4
i
p
y
t
h
o
n
8.3
0.0
i
p
y
t
h
o
n
-
g
e
n
u
t
i
l
s
0.2.0
i
p
y
w
i
d
g
e
t
s
8.1.5
i
s
o
d
u
r
a
t
i
o
n
2
0.1
1.0
i
t
s
d
a
n
g
e
r
o
u
s
2.2.0
j
a
x
0.4.3
4
j
a
x
l
i
b
0.4.3
4
j
a
x
t
y
p
i
n
g
0.2.3
6
j
e
d
i
0.1
9.2
J
i
n
j
a
2
3.1.4
j
i
t
e
r
0.8.2
j
o
b
l
i
b
1.3.2
j
s
o
n
5
0.1
0.0
j
s
o
n
p
a
t
c
h
1.3
3
j
s
o
n
p
a
t
h
-
n
g
1.7.0
j
s
o
n
p
o
i
n
t
e
r
2.1
j
s
o
n
s
c
h
e
m
a
4.2
3.0
j
s
o
n
s
c
h
e
m
a
-
s
p
e
c
i
f
i
c
a
t
i
o
n
s
2
0
2
4.1
0.1
j
u
p
y
t
e
r
1.1.1
j
u
p
y
t
e
r
_
a
i
2.2
8.2
j
u
p
y
t
e
r
_
a
i
_
m
a
g
i
c
s
2.2
8.3
j
u
p
y
t
e
r
_
b
o
k
e
h
4.0.5
j
u
p
y
t
e
r
_
c
l
i
e
n
t
8.6.3
j
u
p
y
t
e
r
-
c
o
n
s
o
l
e
6.6.3
j
u
p
y
t
e
r
_
c
o
r
e
5.7.2
j
u
p
y
t
e
r
-
e
v
e
n
t
s
0.1
0.0
j
u
p
y
t
e
r
-
l
s
p
2.2.5
j
u
p
y
t
e
r
-
r
e
s
o
u
r
c
e
-
u
s
a
g
e
1.1.0
j
u
p
y
t
e
r
_
s
e
r
v
e
r
2.1
4.2
j
u
p
y
t
e
r
_
s
e
r
v
e
r
_
p
r
o
x
y
4.4.0
j
u
p
y
t
e
r
_
s
e
r
v
e
r
_
t
e
r
m
i
n
a
l
s
0.5.3
j
u
p
y
t
e
r
l
a
b
4.3.2
j
u
p
y
t
e
r
l
a
b
_
p
y
g
m
e
n
t
s
0.3.0
j
u
p
y
t
e
r
l
a
b
_
s
e
r
v
e
r
2.2
7.3
j
u
p
y
t
e
r
l
a
b
_
w
i
d
g
e
t
s
3.0.1
3
k
a
g
g
l
e
h
u
b
0.3.4
k
a
l
e
i
d
o
0.2.1
k
e
r
a
s
2.1
4.0
k
e
r
a
s
-
h
u
b
0.1
8.1
k
e
r
a
s
-
n
l
p
0.1
8.1
k
e
r
a
s
-
r
l
0.4.2
k
e
r
a
s
-
t
c
n
3.5.0
k
e
r
a
s
-
t
u
n
e
r
1.4.7
k
i
w
i
s
o
l
v
e
r
1.4.7
k
m
a
p
p
e
r
2.1.0
k
o
r
e
a
n
-
l
u
n
a
r
-
c
a
l
e
n
d
a
r
0.3.1
k
t
-
l
e
g
a
c
y
1.0.5
l
a
n
g
c
h
a
i
n
0.2.1
7
langchain 0.2.17
langchain-community 0.2.19
langchain-core 0.2.43
langchain-text-splitters 0.2.4
langcodes 3.5.0
langsmith 0.1.147
language_data 1.3.0
lark 1.2.2
lazy_loader 0.4
lazypredict 0.2.14a1
libclang 18.1.1
libmambapy 1.5.8
libucx-cu12 1.15.0.post2
lifelines 0.30.0
lightgbm 4.5.0
lightning 2.4.0
lightning-utilities 0.11.9
lime 0.2.0.1
line_profiler 4.2.0
linear-operator 0.5.3
linkify-it-py 2.0.3
livelossplot 0.5.5
llama-cloud 0.1.6
llama-index 0.12.2
llama-index-agent-openai 0.4.0
llama-index-cli 0.4.0
llama-index-core 0.12.5
llama-index-embeddings-openai 0.3.1
llama-index-indices-managed-llama-cloud 0.6.3
llama-index-legacy 0.9.48.post4
llama-index-llms-openai 0.3.3
llama-index-multi-modal-llms-openai 0.3.0
llama-index-program-openai 0.3.1
llama-index-question-gen-openai 0.3.0
llama-index-readers-file 0.4.1
llama-index-readers-llama-parse 0.4.0
llama-parse 0.5.17
llvmlite 0.42.0
locket 1.0.0
logical-unification 0.4.6
loguru 0.7.3
lxml 5.3.0
lz4 4.3.3
Mako 1.3.8
mamba-ssm 2.2.4
MAPIE 0.9.1
marisa-trie 1.2.1
Markdown 3.7
markdown-it-py 3.0.0
MarkupSafe 3.0.2
marshmallow 3.23.1
matplotlib 3.7.5
matplotlib-inline 0.1.7
mdit-py-plugins 0.4.2
mdurl 0.1.2
menuinst 2.1.2
mgarch 0.3.0
miniKanren 1.0.3
minorminer 0.2.15
mistune 3.0.2
ml-dtypes 0.2.0
mlflow 2.18.0
mlflow-skinny 2.18.0
mlforecast 0.15.1
mljar-scikit-plot 0.3.12
mljar-supervised 1.1.9
mlxtend 0.23.3
mmh3 2.5.1
modin 0.26.1
mplfinance 0.12.10b0
mpmath 1.3.0
msgpack 1.1.0
multidict 6.1.0
multipledispatch 1.0.0
multiprocess 0.70.16
multitasking 0.0.11
murmurhash 1.0.11
mypy-extensions 1.0.0
namex 0.0.8
narwhals 1.17.0
nbclient 0.10.1
nbconvert 7.16.4
nbconvert 7.16.4
nbformat 5.10.4
ndindex 1.9.2
nest-asyncio 1.6.0
networkx 3.4.2
neural-tangents 0.6.5
neuralprophet 0.9.0
nfoursid 1.0.1
ngboost 0.5.1
ninja 1.11.1.2
nixtla 0.6.4
nltk 3.9.1
nolds 0.6.1
nose 1.3.7
notebook 7.3.1
notebook_shim 0.2.4
numba 0.59.1
numerapi 2.19.1
numexpr 2.10.2
numpy 1.26.4
nvidia-cublas-cu12 12.1.3.1
nvidia-cuda-cupti-cu12 12.1.105
nvidia-cuda-nvrtc-cu12 12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12 9.1.0.70
nvidia-cufft-cu12 11.0.2.54
nvidia-curand-cu12 10.3.2.106
nvidia-cusolver-cu12 11.4.5.107
nvidia-cusparse-cu12 12.1.0.106
nvidia-nccl-cu12 2.20.5
nvidia-nvjitlink-cu12 12.4.127
nvidia-nvtx-cu12 12.1.105
nvtx 0.2.10
nx-cugraph-cu12 24.8.0
oauthlib 3.2.2
omegaconf 2.3.0
openai 1.57.0
opencv-contrib-python-headless 4.10.0.84
opencv-python 4.10.0.84
openpyxl 3.1.5
opentelemetry-api 1.28.2
opentelemetry-sdk 1.28.2
opentelemetry-semantic-conventions 0.49b2
opt_einsum 3.4.0
optree 0.13.1
optuna 4.1.0
orjson 3.10.12
ortools 9.9.3963
osqp 0.6.7.post3
overrides 7.7.0
packaging 24.1
pandas 2.1.4
pandas-flavor 0.6.0
pandas_market_calendars 4.4.2
pandas_ta 0.3.14b0
pandocfilters 1.5.1
panel 1.5.4
param 2.1.1
parso 0.8.4
partd 1.4.2
pastel 0.2.1
pathos 0.3.2
patsy 1.0.1
pbr 6.1.0
pearl 2.3.12
peewee 3.17.3
peft 0.13.2
penaltymodel 1.1.0
PennyLane 0.39.0
PennyLane_Lightning 0.39.0
PennyLane-qiskit 0.36.0
persim 0.3.7
pexpect 4.9.0
pgmpy 0.1.26
pillow 10.4.0
pingouin 0.5.5
pip 24.0
platformdirs 3.10.0
plotly 5.24.1
plotly-resampler 0.10.0
plucky 0.4.3
pluggy 1.5.0
p
l
u
g
g
y
1.5.0
p
l
y
3.1
1
p
m
d
a
r
i
m
a
2.0.4
p
o
l
a
r
s
0.2
0.3
1
p
o
m
e
g
r
a
n
a
t
e
1.1.1
P
O
T
0.9.5
p
o
x
0.3.5
p
p
f
t
1.7.6.9
p
p
r
o
f
i
l
e
2.2.0
p
r
e
s
h
e
d
3.0.9
p
r
o
m
e
t
h
e
u
s
_
c
l
i
e
n
t
0.2
1.1
p
r
o
m
p
t
_
t
o
o
l
k
i
t
3.0.4
8
p
r
o
p
c
a
c
h
e
0.2.1
p
r
o
p
h
e
t
1.1.6
p
r
o
t
o
-
p
l
u
s
1.2
5.0
p
r
o
t
o
b
u
f
4.2
5.5
p
s
u
t
i
l
5.9.8
p
t
y
p
r
o
c
e
s
s
0.7.0
P
u
L
P
2.9.0
p
u
r
e
_
e
v
a
l
0.2.3
p
y
-
c
p
u
i
n
f
o
9.0.0
p
y
-
h
e
a
t
0.0.6
p
y
-
h
e
a
t
-
m
a
g
i
c
0.0.2
p
y
_
l
e
t
s
_
b
e
_
r
a
t
i
o
n
a
l
1.0.1
p
y
_
v
o
l
l
i
b
1.0.1
p
y
4
j
0.1
0.9.7
p
y
a
m
l
2
4.9.0
p
y
a
r
r
o
w
1
6.1.0
p
y
a
r
r
o
w
-
h
o
t
f
i
x
0.6
p
y
a
s
n
1
0.6.1
p
y
a
s
n
1
_
m
o
d
u
l
e
s
0.4.1
p
y
b
i
n
d
1
1
2.1
3.6
p
y
c
a
r
e
t
3.3.2
p
y
c
o
s
a
t
0.6.6
p
y
c
p
a
r
s
e
r
2.2
1
p
y
c
t
0.5.0
p
y
d
a
n
t
i
c
2.9.2
p
y
d
a
n
t
i
c
_
c
o
r
e
2.2
3.4
P
y
D
M
D
2
0
2
4.1
2.1
p
y
e
r
f
a
2.0.1.5
p
y
f
o
l
i
o
-
r
e
l
o
a
d
e
d
0.9.8
P
y
g
m
e
n
t
s
2.1
8.0
P
y
J
W
T
2.1
0.1
p
y
k
a
l
m
a
n
0.9.7
p
y
l
e
v
1.4.0
p
y
l
i
b
c
u
g
r
a
p
h
-
c
u
1
2
2
4.8.0
p
y
l
i
b
r
a
f
t
-
c
u
1
2
2
4.8.1
p
y
l
u
a
c
h
2.2.0
p
y
m
a
n
n
k
e
n
d
a
l
l
1.4.3
p
y
m
c
5.1
9.0
p
y
m
d
p
t
o
o
l
b
o
x
4.0
b
3
p
y
n
n
d
e
s
c
e
n
t
0.5.1
3
p
y
n
v
j
i
t
l
i
n
k
-
c
u
1
2
0.4.0
p
y
n
v
m
l
1
1.4.1
p
y
o
d
2.0.2
p
y
o
g
r
i
o
0.1
0.0
P
y
o
m
o
6.8.2
p
y
p
a
r
s
i
n
g
3.2.0
p
y
p
d
f
5.1.0
p
y
p
o
r
t
f
o
l
i
o
o
p
t
1.5.6
p
y
p
r
o
j
3.7.0
P
y
Q
t
6
6.7.1
P
y
Q
t
6
-
Q
t
6
6.7.3
P
y
Q
t
6
_
s
i
p
1
3.9.0
p
y
r
b
1.0.1
p
y
r
e
-
e
x
t
e
n
s
i
o
n
s
0.0.3
2
p
y
r
o
-
a
p
i
0.1.2
p
y
r
o
-
p
p
l
1.9.1
p
y
s
i
m
d
j
s
o
n
6.0.2
P
y
S
o
c
k
s
1.7.1
p
y
s
p
n
e
g
o
0.1
1.2
p
y
s
t
a
n
3.1
0.0
p
y
t
e
n
s
o
r
2.2
6.4
p
y
t
e
s
t
8.3.4
p
y
t
e
s
t
-
r
u
n
n
e
r
6.0.1
p
y
t
h
o
n
-
d
a
t
e
u
t
i
l
2.9.0.p
o
s
t
0
p
y
t
h
o
n
-
d
o
t
e
n
v
1.0.0
p
y
t
h
o
n
-
j
s
o
n
-
l
o
g
g
e
r
2.0.7
p
y
t
h
o
n
-
s
t
a
t
e
m
a
c
h
i
n
e
2.5.0
p
y
t
o
r
c
h
-
f
o
r
e
c
a
s
t
i
n
g
1.2.0
p
y
t
o
r
c
h
-
i
g
n
i
t
e
0.5.1
p
y
t
o
r
c
h
-
l
i
g
h
t
n
i
n
g
2.4.0
p
y
t
o
r
c
h
-
l
i
g
h
t
n
i
n
g
2.4.0
p
y
t
o
r
c
h
-
t
a
b
n
e
t
4.1.0
p
y
t
z
2
0
2
4.2
p
y
v
i
n
e
c
o
p
u
l
i
b
0.6.5
p
y
v
i
z
_
c
o
m
m
s
3.0.3
P
y
W
a
v
e
l
e
t
s
1.7.0
P
y
Y
A
M
L
6.0.2
p
y
z
m
q
2
6.2.0
q
d
l
d
l
0.1.7.p
o
s
t
4
q
i
s
k
i
t
1.2.4
q
i
s
k
i
t
-
a
e
r
0.1
5.1
q
i
s
k
i
t
-
i
b
m
-
p
r
o
v
i
d
e
r
0.1
1.0
q
i
s
k
i
t
-
i
b
m
-
r
u
n
t
i
m
e
0.3
4.0
q
u
a
d
p
r
o
g
0.1.1
3
q
u
a
n
t
e
c
o
n
0.7.2
Q
u
a
n
t
L
i
b
1.3
6
Q
u
a
n
t
S
t
a
t
s
0.0.6
4
r
a
f
t
-
d
a
s
k
-
c
u
1
2
2
4.8.1
r
a
p
i
d
s
-
d
a
s
k
-
d
e
p
e
n
d
e
n
c
y
2
4.8.0
r
a
u
t
h
0.7.3
r
a
y
2.4
0.0
R
b
e
a
s
t
0.1.2
3
r
e
f
e
r
e
n
c
i
n
g
0.3
5.1
r
e
g
e
x
2
0
2
4.1
1.6
r
e
q
u
e
s
t
s
2.3
2.3
r
e
q
u
e
s
t
s
_
n
t
l
m
1.3.0
r
e
q
u
e
s
t
s
-
o
a
u
t
h
l
i
b
1.3.1
r
e
q
u
e
s
t
s
-
t
o
o
l
b
e
l
t
1.0.0
r
f
c
3
3
3
9
-
v
a
l
i
d
a
t
o
r
0.1.4
r
f
c
3
9
8
6
-
v
a
l
i
d
a
t
o
r
0.1.1
r
i
c
h
1
3.9.4
r
i
p
s
e
r
0.6.1
0
R
i
s
k
f
o
l
i
o
-
L
i
b
6.1.1
r
i
s
k
p
a
r
i
t
y
p
o
r
t
f
o
l
i
o
0.6.0
r
i
v
e
r
0.2
1.0
r
m
m
-
c
u
1
2
2
4.8.2
r
p
d
s
-
p
y
0.2
2.3
r
s
a
4.9
r
u
a
m
e
l.y
a
m
l
0.1
8.6
r
u
a
m
e
l.y
a
m
l.c
l
i
b
0.2.8
r
u
p
t
u
r
e
s
1.1.9
r
u
s
t
w
o
r
k
x
0.1
5.1
s
a
f
e
t
e
n
s
o
r
s
0.4.5
S
A
L
i
b
1.5.1
s
c
h
e
m
d
r
a
w
0.1
5
s
c
i
k
e
r
a
s
0.1
3.0
s
c
i
k
i
t
-
b
a
s
e
0.7.8
s
c
i
k
i
t
-
i
m
a
g
e
0.2
2.0
s
c
i
k
i
t
-
l
e
a
r
n
1.4.2
s
c
i
k
i
t
-
l
e
a
r
n
-
e
x
t
r
a
0.3.0
s
c
i
k
i
t
-
o
p
t
i
m
i
z
e
0.1
0.2
s
c
i
k
i
t
-
p
l
o
t
0.3.7
s
c
i
k
i
t
-
t
d
a
1.1.1
s
c
i
p
y
1.1
1.4
s
c
s
3.2.7
s
d
e
i
n
t
0.3.0
s
e
a
b
o
r
n
0.1
3.2
S
e
n
d
2
T
r
a
s
h
1.8.3
s
e
n
t
e
n
c
e
-
t
r
a
n
s
f
o
r
m
e
r
s
3.3.1
s
e
t
u
p
t
o
o
l
s
6
5.5.0
s
e
t
u
p
t
o
o
l
s
-
s
c
m
8.1.0
s
h
a
p
0.4
6.0
s
h
a
p
e
l
y
2.0.6
S
h
i
m
m
y
2.0.0
s
i
m
p
e
r
v
i
s
o
r
1.0.0
s
i
m
p
l
e
j
s
o
n
3.1
9.3
s
i
m
p
y
4.1.1
s
i
x
1.1
7.0
s
k
l
e
a
r
n
-
j
s
o
n
0.1.0
s
k
t
i
m
e
0.2
6.0
s
l
i
c
e
r
0.0.8
s
m
a
r
t
-
o
p
e
n
7.0.5
s
m
m
a
p
5.0.1
s
n
i
f
f
i
o
1.3.1
s
o
r
t
e
d
c
o
n
t
a
i
n
e
r
s
2.4.0
s
o
u
p
s
i
e
v
e
2.6
s
p
a
c
y
3.7.5
s
p
a
c
y
-
l
e
g
a
c
y
3.0.1
2
s
p
a
c
y
-
l
o
g
g
e
r
s
1.0.5
S
Q
L
A
l
c
h
e
m
y
2.0.3
6
s
q
l
p
a
r
s
e
0.5.3
s
r
s
l
y
2.5.0
srsly 2.5.0
ssm 0.0.1
stable_baselines3 2.4.0
stack-data 0.6.3
stanio 0.5.1
statsforecast 2.0.0
statsmodels 0.14.4
stevedore 5.4.0
stochastic 0.6.0
stockstats 0.6.2
stopit 1.1.2
striprtf 0.0.26
stumpy 1.13.0
symengine 0.13.0
sympy 1.13.1
ta 0.11.0
ta-lib 0.5.1
tables 3.10.1
tabulate 0.8.10
tadasets 0.2.1
tbats 1.1.3
tblib 3.0.0
tenacity 8.5.0
tensorboard 2.14.1
tensorboard-data-server 0.7.2
tensorboardX 2.6.2.2
tensorflow 2.14.1
tensorflow-addons 0.23.0
tensorflow_decision_forests 1.11.0
tensorflow-estimator 2.14.0
tensorflow-io-gcs-filesystem 0.37.1
tensorflow-probability 0.25.0
tensorflow-ranking 0.5.3
tensorflow-serving-api 2.14.1
tensorflow-text 2.18.0
tensorly 0.9.0
tensorrt 10.7.0
tensorrt_cu12 10.7.0
tensorrt-cu12-bindings 10.7.0
tensorrt-cu12-libs 10.7.0
tensortrade 1.0.3
termcolor 2.5.0
terminado 0.18.1
tf_keras 2.18.0
tf2jax 0.3.6
thinc 8.2.5
threadpoolctl 3.5.0
thundergbm 0.3.17
tifffile 2024.9.20
tigramite 5.2.6.7
tiktoken 0.8.0
tinycss2 1.4.0
tinygrad 0.10.0
tokenizers 0.20.3
toml 0.10.2
toolz 0.12.1
torch 2.4.1
torch_cluster 1.6.3
torch-geometric 2.6.1
torch_scatter 2.1.2
torch_sparse 0.6.18
torch_spline_conv 1.2.2
torchdata 0.10.0
torchmetrics 1.6.0
torchvision 0.20.1
tornado 6.4.2
TPOT 0.12.2
tqdm 4.66.5
traitlets 5.14.3
transformers 4.46.3
treelite 4.3.0
triad 0.9.8
triton 3.0.0
truststore 0.8.0
tsdownsample 0.1.3
tsfel 0.1.9
tsfresh 0.20.2
tslearn 0.6.3
tweepy 4.14.0
typeguard 2.13.3
typer 0.9.4
typer-config 1.4.2
A
u
t
o
glu
o
n
E
n
vir
o
n
m
e
n
t
T
h
e
A
u
t
o
glu
o
n
e
n
vir
o
n
m
e
n
t
p
r
o
vid
e
s
t
h
e
f
ollo
win
g lib
r
a
rie
s: typer-config 1.4.2 types-python-dateutil 2.9.0.20241206 typing_extensions 4.12.2 typing-inspect 0.9.0 tzdata 2024.2 uc-micro-py 1.0.3 ucx-py-cu12 0.39.2 ucxx-cu12 0.39.1 umap-learn 0.5.7 uni2ts 1.2.0 update-checker 0.18.0 uri-template 1.3.0 uritemplate 4.1.1 urllib3 2.2.3 utilsforecast 0.2.10 wasabi 1.1.3 wcwidth 0.2.13 weasel 0.4.1 webargs 8.6.0 webcolors 24.11.1 webencodings 0.5.1 websocket-client 1.8.0 websockets 14.1 Werkzeug 3.1.3 wheel 0.44.0 widgetsnbextension 4.0.13 window_ops 0.0.15 woodwork 0.31.0 wordcloud 1.9.4 wrapt 1.14.1 wurlitzer 3.1.1 x-transformers 1.42.24 xarray 2024.11.0 xarray-einstats 0.8.0 xgboost 2.1.3 xlrd 2.0.1 XlsxWriter 3.2.0 xxhash 3.5.0 xyzservices 2024.9.0 yarl 1.18.3 ydf 0.9.0 yellowbrick 1.5 yfinance 0.2.50 zict 3.0.0 zipp 3.21.0 zope.event 5.0 zope.interface 7.2 zstandard 0.23.0 absl-py 2.1.0 accelerate 0.34.2 adagio 0.2.6 aesara 2.9.4 aiohappyeyeballs 2.4.4 aiohttp 3.11.10 aiohttp-cors 0.7.0 aiosignal 1.3.1 aiosqlite 0.20.0 alembic 1.14.0 alibi-detect 0.12.0 alphalens-reloaded 0.4.5 altair 5.5.0 anaconda-anon-usage 0.4.4 annotated-types 0.7.0 antlr4-python3-runtime 4.9.3 anyio 4.7.0 aplr 10.8.0 appdirs 1.4.4 apricot-select 0.6.1 arch 7.2.0 archspec 0.2.3 argon2-cffi 23.1.0 argon2-cffi-bindings 21.2.0
P
Y
a
r
g
o
n
2
-
c
f
f
i
-
b
i
n
d
i
n
g
s
2
1.2.0
a
r
r
o
w
1.3.0
a
r
v
i
z
0.2
0.0
a
s
t
r
o
p
y
7.0.0
a
s
t
r
o
p
y
-
i
e
r
s
-
d
a
t
a
0.2
0
2
4.1
2.9.0.3
6.2
1
a
s
t
t
o
k
e
n
s
3.0.0
a
s
t
u
n
p
a
r
s
e
1.6.3
a
s
y
n
c
-
l
r
u
2.0.4
a
t
t
r
s
2
4.2.0
A
u
t
h
l
i
b
1.3.2
a
u
t
o
g
l
u
o
n
1.2
a
u
t
o
g
l
u
o
n.c
o
m
m
o
n
1.2
a
u
t
o
g
l
u
o
n.c
o
r
e
1.2
a
u
t
o
g
l
u
o
n.f
e
a
t
u
r
e
s
1.2
a
u
t
o
g
l
u
o
n.m
u
l
t
i
m
o
d
a
l
1.2
a
u
t
o
g
l
u
o
n.t
a
b
u
l
a
r
1.2
a
u
t
o
g
l
u
o
n.t
i
m
e
s
e
r
i
e
s
1.2
a
u
t
o
g
r
a
d
1.7.0
a
u
t
o
g
r
a
d
-
g
a
m
m
a
0.5.0
a
u
t
o
k
e
r
a
s
2.0.0
a
u
t
o
r
a
y
0.7.0
a
x
-
p
l
a
t
f
o
r
m
0.4.3
b
a
b
e
l
2.1
6.0
b
a
y
e
s
i
a
n
-
o
p
t
i
m
i
z
a
t
i
o
n
2.0.0
b
e
a
u
t
i
f
u
l
s
o
u
p
4
4.1
2.3
b
l
e
a
c
h
6.2.0
b
l
i
n
k
e
r
1.9.0
b
l
i
s
0.7.1
1
b
l
o
s
c
2
2.7.1
b
o
k
e
h
3.6.2
b
o
l
t
o
n
s
2
3.0.0
b
o
t
o
3
1.3
5.8
6
b
o
t
o
c
o
r
e
1.3
5.8
6
b
o
t
o
r
c
h
0.1
2.0
B
o
t
t
l
e
n
e
c
k
1.4.2
B
r
o
t
l
i
1.0.9
c
a
c
h
e
t
o
o
l
s
5.5.0
c
a
p
t
u
m
0.7.0
c
a
t
a
l
o
g
u
e
2.0.1
0
c
a
t
b
o
o
s
t
1.2.7
c
a
t
e
g
o
r
y
-
e
n
c
o
d
e
r
s
2.6.4
c
a
u
s
a
l
-
c
o
n
v
1
d
1.5.0.p
o
s
t
8
c
e
r
t
i
f
i
2
0
2
4.8.3
0
c
e
s
i
u
m
0.1
2.1
c
f
f
i
1.1
7.1
c
h
a
r
d
e
t
5.2.0
c
h
a
r
s
e
t
-
n
o
r
m
a
l
i
z
e
r
3.3.2
c
h
e
c
k
-
s
h
a
p
e
s
1.1.1
c
h
r
o
n
o
s
-
f
o
r
e
c
a
s
t
i
n
g
1.4.1
c
l
a
r
a
b
e
l
0.9.0
c
l
i
c
k
8.1.7
c
l
i
k
i
t
0.6.2
c
l
o
u
d
p
a
t
h
l
i
b
0.2
0.0
c
l
o
u
d
p
i
c
k
l
e
3.1.0
c
m
d
s
t
a
n
p
y
1.2.4
c
o
l
o
r
a
m
a
0.4.6
c
o
l
o
r
c
e
t
3.1.0
c
o
l
o
r
f
u
l
0.5.6
c
o
l
o
r
l
o
g
6.9.0
c
o
l
o
r
l
o
v
e
r
0.3.0
c
o
l
o
u
r
0.1.5
c
o
m
m
0.2.2
c
o
n
d
a
2
4.9.2
c
o
n
d
a
-
c
o
n
t
e
n
t
-
t
r
u
s
t
0.2.0
c
o
n
d
a
-
l
i
b
m
a
m
b
a
-
s
o
l
v
e
r
2
4.9.0
c
o
n
d
a
-
p
a
c
k
a
g
e
-
h
a
n
d
l
i
n
g
2.3.0
c
o
n
d
a
_
p
a
c
k
a
g
e
_
s
t
r
e
a
m
i
n
g
0.1
0.0
c
o
n
f
e
c
t
i
o
n
0.1.5
c
o
n
s
0.4.6
c
o
n
t
o
u
r
p
y
1.3.1
c
o
n
t
r
o
l
0.1
0.1
c
o
p
u
l
a
e
0.7.9
c
o
p
u
l
a
s
0.1
2.0
c
o
r
e
f
o
r
e
c
a
s
t
0.0.1
2
c
r
a
m
j
a
m
2.9.0
c
r
a
s
h
t
e
s
t
0.3.1
c
r
e
m
e
0.6.1
c
r
y
p
t
o
g
r
a
p
h
y
4
3.0.0
c
u
c
i
m
-
c
u
1
2
2
4.8.0
c
u
d
a
-
p
y
t
h
o
n
1
2.6.2.p
o
s
t
1
c
u
d
f
-
c
u
1
2
2
4.8.3
c
u
f
f
l
i
n
k
s
0.1
7.3
c
u
f
f
l
i
n
k
s
0.1
7.3
c
u
g
r
a
p
h
-
c
u
1
2
2
4.8.0
c
u
m
l
-
c
u
1
2
2
4.8.0
c
u
p
r
o
j
-
c
u
1
2
2
4.8.0
c
u
p
y
-
c
u
d
a
1
2
x
1
3.3.0
c
u
s
p
a
t
i
a
l
-
c
u
1
2
2
4.8.0
c
u
v
s
-
c
u
1
2
2
4.8.0
c
u
x
f
i
l
t
e
r
-
c
u
1
2
2
4.8.0
c
v
x
o
p
t
1.3.2
c
v
x
p
o
r
t
f
o
l
i
o
1.4.0
c
v
x
p
y
1.6.0
c
y
c
l
e
r
0.1
2.1
c
y
m
e
m
2.0.1
0
C
y
t
h
o
n
3.0.1
1
d
a
r
t
s
0.3
1.0
d
a
s
h
2.9.3
d
a
s
h
-
c
o
r
e
-
c
o
m
p
o
n
e
n
t
s
2.0.0
d
a
s
h
_
c
y
t
o
s
c
a
p
e
1.0.2
d
a
s
h
-
h
t
m
l
-
c
o
m
p
o
n
e
n
t
s
2.0.0
d
a
s
h
-
t
a
b
l
e
5.0.0
d
a
s
k
2
0
2
4.7.1
d
a
s
k
-
c
u
d
a
2
4.8.2
d
a
s
k
-
c
u
d
f
-
c
u
1
2
2
4.8.3
d
a
s
k
-
e
x
p
r
1.1.9
d
a
t
a
b
r
i
c
k
s
-
s
d
k
0.3
8.0
d
a
t
a
c
l
a
s
s
e
s
-
j
s
o
n
0.6.7
d
a
t
a
s
e
t
s
2.2
1.0
d
a
t
a
s
h
a
d
e
r
0.1
6.3
d
e
a
p
1.4.1
d
e
b
u
g
p
y
1.8.9
d
e
c
o
r
a
t
o
r
5.1.1
d
e
e
p
m
e
r
g
e
2.0
d
e
f
u
s
e
d
x
m
l
0.7.1
D
e
p
r
e
c
a
t
e
d
1.2.1
5
d
e
p
r
e
c
a
t
i
o
n
2.1.0
d
g
l
2.1.0
d
i
l
l
0.3.8
d
i
m
o
d
0.1
2.1
7
d
i
r
t
y
j
s
o
n
1.0.8
d
i
s
k
c
a
c
h
e
5.6.3
d
i
s
t
l
i
b
0.3.9
d
i
s
t
r
i
b
u
t
e
d
2
0
2
4.7.1
d
i
s
t
r
i
b
u
t
e
d
-
u
c
x
x
-
c
u
1
2
0.3
9.1
d
i
s
t
r
o
1.9.0
d
m
-
t
r
e
e
0.1.8
d
o
c
k
e
r
7.1.0
d
o
c
u
t
i
l
s
0.2
1.2
D
o
u
b
l
e
M
L
0.9.0
d
r
o
p
s
t
a
c
k
f
r
a
m
e
0.1.1
d
t
r
e
e
v
i
z
2.2.2
d
t
w
-
p
y
t
h
o
n
1.5.3
d
w
a
v
e
-
c
l
o
u
d
-
c
l
i
e
n
t
0.1
3.1
d
w
a
v
e
-
d
r
i
v
e
r
s
0.4.4
d
w
a
v
e
-
g
a
t
e
0.3.2
d
w
a
v
e
-
g
r
e
e
d
y
0.3.0
d
w
a
v
e
-
h
y
b
r
i
d
0.6.1
2
d
w
a
v
e
-
i
n
s
p
e
c
t
o
r
0.5.1
d
w
a
v
e
-
i
n
s
p
e
c
t
o
r
a
p
p
0.3.3
d
w
a
v
e
-
n
e
a
l
0.6.0
d
w
a
v
e
_
n
e
t
w
o
r
k
x
0.8.1
5
d
w
a
v
e
-
o
c
e
a
n
-
s
d
k
8.0.1
d
w
a
v
e
-
o
p
t
i
m
i
z
a
t
i
o
n
0.3.0
d
w
a
v
e
-
p
r
e
p
r
o
c
e
s
s
i
n
g
0.6.6
d
w
a
v
e
-
s
a
m
p
l
e
r
s
1.3.0
d
w
a
v
e
-
s
y
s
t
e
m
1.2
6.0
d
w
a
v
e
-
t
a
b
u
0.5.0
d
w
a
v
e
b
i
n
a
r
y
c
s
p
0.3.0
e
c
o
s
2.0.1
4
e
i
n
o
p
s
0.8.0
e
i
n
x
0.3.0
E
M
D
-
s
i
g
n
a
l
1.6.4
e
m
p
y
r
i
c
a
l
-
r
e
l
o
a
d
e
d
0.5.1
1
e
n
-
c
o
r
e
-
w
e
b
-
m
d
3.7.1
e
n
-
c
o
r
e
-
w
e
b
-
s
m
3.7.1
e
t
_
x
m
l
f
i
l
e
2.0.0
e
t
u
p
l
e
s
0.3.9
e
v
a
l
u
a
t
e
0.4.3
e
x
c
h
a
n
g
e
_
c
a
l
e
n
d
a
r
s
4.6
e
x
e
c
u
t
i
n
g
2.1.0
f
a
i
s
s
-
c
p
u
1.9.0.p
o
s
t
1
F
a
r
a
m
a
-
N
o
t
i
f
i
c
a
t
i
o
n
s
0.0.4
f
a
s
t
a
i
2.7.1
8
f
a
s
t
a
i
2.7.1
8
f
a
s
t
a
i
2
0.0.3
0
f
a
s
t
c
o
r
e
1.7.2
6
f
a
s
t
d
o
w
n
l
o
a
d
0.0.7
f
a
s
t
e
n
e
r
s
0.1
9
f
a
s
t
j
s
o
n
s
c
h
e
m
a
2.2
1.1
f
a
s
t
p
a
r
q
u
e
t
2
0
2
4.1
1.0
f
a
s
t
p
r
o
g
r
e
s
s
1.0.3
f
a
s
t
r
l
o
c
k
0.8.2
f
a
s
t
t
e
x
t
0.9.3
f
e
a
t
u
r
e
-
e
n
g
i
n
e
1.6.2
f
e
a
t
u
r
e
t
o
o
l
s
1.3
1.0
f
i
l
e
l
o
c
k
3.1
6.1
f
i
l
e
t
y
p
e
1.2.0
f
i
n
d
i
f
f
0.1
0.2
F
i
x
e
d
E
f
f
e
c
t
M
o
d
e
l
0.0.5
F
l
a
g
E
m
b
e
d
d
i
n
g
1.2.1
1
F
l
a
s
k
3.1.0
f
l
a
t
b
u
f
f
e
r
s
2
4.3.2
5
f
o
n
t
t
o
o
l
s
4.5
5.2
f
o
r
m
u
l
a
i
c
1.0.2
f
q
d
n
1.5.1
f
r
o
z
e
n
d
i
c
t
2.4.2
f
r
o
z
e
n
l
i
s
t
1.5.0
f
s
2.4.1
6
f
s
s
p
e
c
2
0
2
4.6.1
f
u
g
u
e
0.9.1
f
u
t
u
r
e
1.0.0
f
u
z
z
y
-
c
-
m
e
a
n
s
1.7.2
g
a
s
t
0.6.0
g
a
t
s
p
y
0.3
g
d
o
w
n
5.2.0
g
e
n
s
i
m
4.3.3
g
e
o
p
a
n
d
a
s
1.0.1
g
e
v
e
n
t
2
4.1
1.1
g
i
t
d
b
4.0.1
1
G
i
t
P
y
t
h
o
n
3.1.4
3
g
l
u
o
n
t
s
0.1
6.0
g
o
o
g
l
e
-
a
i
-
g
e
n
e
r
a
t
i
v
e
l
a
n
g
u
a
g
e
0.6.1
0
g
o
o
g
l
e
-
a
p
i
-
c
o
r
e
2.2
4.0
g
o
o
g
l
e
-
a
p
i
-
p
y
t
h
o
n
-
c
l
i
e
n
t
2.1
5
4.0
g
o
o
g
l
e
-
a
u
t
h
2.3
6.0
g
o
o
g
l
e
-
a
u
t
h
-
h
t
t
p
l
i
b
2
0.2.0
g
o
o
g
l
e
-
g
e
n
e
r
a
t
i
v
e
a
i
0.8.3
g
o
o
g
l
e
-
p
a
s
t
a
0.2.0
g
o
o
g
l
e
a
p
i
s
-
c
o
m
m
o
n
-
p
r
o
t
o
s
1.6
6.0
g
p
f
l
o
w
2.9.2
g
p
l
e
a
r
n
0.4.2
g
p
y
t
o
r
c
h
1.1
3
g
r
a
p
h
e
n
e
3.4.3
g
r
a
p
h
q
l
-
c
o
r
e
3.2.5
g
r
a
p
h
q
l
-
r
e
l
a
y
3.2.0
g
r
a
p
h
v
i
z
0.2
0.3
g
r
e
e
n
l
e
t
3.1.1
g
r
p
c
i
o
1.6
8.1
g
r
p
c
i
o
-
s
t
a
t
u
s
1.6
8.1
g
u
n
i
c
o
r
n
2
3.0.0
g
y
m
0.2
6.2
g
y
m
-
n
o
t
i
c
e
s
0.0.8
g
y
m
n
a
s
i
u
m
1.0.0
h
1
1
0.1
4.0
h
2
o
3.4
6.0.6
h
5
n
e
t
c
d
f
1.4.1
h
5
p
y
3.1
2.1
h
m
m
l
e
a
r
n
0.3.3
h
o
l
i
d
a
y
s
0.6
2
h
o
l
o
v
i
e
w
s
1.2
0.0
h
o
m
e
b
a
s
e
1.0.1
h
o
p
c
r
o
f
t
k
a
r
p
1.2.5
h
t
m
l
5
l
i
b
1.1
h
t
t
p
c
o
r
e
1.0.7
h
t
t
p
l
i
b
2
0.2
2.0
h
t
t
p
s
t
a
n
4.1
3.0
h
t
t
p
x
0.2
8.1
h
u
g
g
i
n
g
f
a
c
e
-
h
u
b
0.2
6.5
h
u
r
s
t
0.0.5
h
v
p
l
o
t
0.1
1.1
h
y
p
e
r
o
p
t
0.2.7
i
b
m
-
c
l
o
u
d
-
s
d
k
-
c
o
r
e
3.2
2.0
i
b
m
-
p
l
a
t
f
o
r
m
-
s
e
r
v
i
c
e
s
0.5
9.0
i
d
n
a
3.7
i
i
s
i
g
n
a
t
u
r
e
0.2
4
i
i
s
i
g
n
a
t
u
r
e
0.2
4
i
j
s
o
n
3.3.0
i
m
a
g
e
i
o
2.3
6.1
i
m
b
a
l
a
n
c
e
d
-
l
e
a
r
n
0.1
2.4
i
m
m
u
t
a
b
l
e
d
i
c
t
4.2.1
i
m
p
o
r
t
l
i
b
_
m
e
t
a
d
a
t
a
8.5.0
i
m
p
o
r
t
l
i
b
_
r
e
s
o
u
r
c
e
s
6.4.5
i
n
i
c
o
n
f
i
g
2.0.0
i
n
j
e
c
t
o
r
0.2
2.0
i
n
t
e
r
f
a
c
e
-
m
e
t
a
1.3.0
i
n
t
e
r
p
r
e
t
0.6.7
i
n
t
e
r
p
r
e
t
-
c
o
r
e
0.6.7
i
p
y
k
e
r
n
e
l
6.2
9.5
i
p
y
m
p
l
0.9.4
i
p
y
t
h
o
n
8.3
0.0
i
p
y
t
h
o
n
-
g
e
n
u
t
i
l
s
0.2.0
i
p
y
w
i
d
g
e
t
s
8.1.5
i
s
o
d
u
r
a
t
i
o
n
2
0.1
1.0
i
t
s
d
a
n
g
e
r
o
u
s
2.2.0
j
a
x
0.4.3
5
j
a
x
l
i
b
0.4.3
5
j
a
x
t
y
p
i
n
g
0.2.1
9
j
e
d
i
0.1
9.2
J
i
n
j
a
2
3.1.4
j
i
t
e
r
0.8.2
j
m
e
s
p
a
t
h
1.0.1
j
o
b
l
i
b
1.3.2
j
s
o
n
5
0.1
0.0
j
s
o
n
p
a
t
c
h
1.3
3
j
s
o
n
p
a
t
h
-
n
g
1.7.0
j
s
o
n
p
o
i
n
t
e
r
2.1
j
s
o
n
s
c
h
e
m
a
4.2
1.1
j
s
o
n
s
c
h
e
m
a
-
s
p
e
c
i
f
i
c
a
t
i
o
n
s
2
0
2
4.1
0.1
j
u
p
y
t
e
r
1.1.1
j
u
p
y
t
e
r
_
a
i
2.2
8.2
j
u
p
y
t
e
r
_
a
i
_
m
a
g
i
c
s
2.2
8.3
j
u
p
y
t
e
r
_
b
o
k
e
h
4.0.5
j
u
p
y
t
e
r
_
c
l
i
e
n
t
8.6.3
j
u
p
y
t
e
r
-
c
o
n
s
o
l
e
6.6.3
j
u
p
y
t
e
r
_
c
o
r
e
5.7.2
j
u
p
y
t
e
r
-
e
v
e
n
t
s
0.1
0.0
j
u
p
y
t
e
r
-
l
s
p
2.2.5
j
u
p
y
t
e
r
-
r
e
s
o
u
r
c
e
-
u
s
a
g
e
1.1.0
j
u
p
y
t
e
r
_
s
e
r
v
e
r
2.1
4.2
j
u
p
y
t
e
r
_
s
e
r
v
e
r
_
p
r
o
x
y
4.4.0
j
u
p
y
t
e
r
_
s
e
r
v
e
r
_
t
e
r
m
i
n
a
l
s
0.5.3
j
u
p
y
t
e
r
l
a
b
4.3.2
j
u
p
y
t
e
r
l
a
b
_
p
y
g
m
e
n
t
s
0.3.0
j
u
p
y
t
e
r
l
a
b
_
s
e
r
v
e
r
2.2
7.3
j
u
p
y
t
e
r
l
a
b
_
w
i
d
g
e
t
s
3.0.1
3
k
a
g
g
l
e
h
u
b
0.3.4
k
a
l
e
i
d
o
0.2.1
k
e
r
a
s
3.7.0
k
e
r
a
s
-
h
u
b
0.1
8.1
k
e
r
a
s
-
n
l
p
0.1
8.1
k
e
r
a
s
-
r
l
0.4.2
k
e
r
a
s
-
t
c
n
3.5.0
k
e
r
a
s
-
t
u
n
e
r
1.4.7
k
i
w
i
s
o
l
v
e
r
1.4.7
k
m
a
p
p
e
r
2.1.0
k
o
r
e
a
n
-
l
u
n
a
r
-
c
a
l
e
n
d
a
r
0.3.1
k
t
-
l
e
g
a
c
y
1.0.5
l
a
n
g
c
h
a
i
n
0.2.1
7
l
a
n
g
c
h
a
i
n
-
c
o
m
m
u
n
i
t
y
0.2.1
9
l
a
n
g
c
h
a
i
n
-
c
o
r
e
0.2.4
3
l
a
n
g
c
h
a
i
n
-
t
e
x
t
-
s
p
l
i
t
t
e
r
s
0.2.4
l
a
n
g
c
o
d
e
s
3.5.0
l
a
n
g
s
m
i
t
h
0.1.1
4
7
l
a
n
g
u
a
g
e
_
d
a
t
a
1.3.0
l
a
r
k
1.2.2
l
a
z
y
_
l
o
a
d
e
r
0.4
l
a
z
y
p
r
e
d
i
c
t
0.2.1
4
a
1
l
i
b
c
l
a
n
g
1
8.1.1
l
i
b
m
a
m
b
a
p
y
1.5.8
l
i
b
u
c
x
-
c
u
1
2
1.1
5.0.p
o
s
t
2
l
i
f
e
l
i
n
e
s
0.3
0.0
l
i
g
h
t
g
b
m
4.5.0
l
i
g
h
t
n
i
n
g
2.4.0
l
i
g
h
t
n
i
n
g
-
u
t
i
l
i
t
i
e
s
0.1
1.9
l
i
m
e
0.2.0.1
l
i
n
e
_
p
r
o
f
i
l
e
r
4.2.0
l
i
n
e
a
r
-
o
p
e
r
a
t
o
r
0.5.3
linear-operator 0.5.3
linkify-it-py 2.0.3
livelossplot 0.5.5
llama-cloud 0.1.6
llama-index 0.12.2
llama-index-agent-openai 0.4.0
llama-index-cli 0.4.0
llama-index-core 0.12.5
llama-index-embeddings-openai 0.3.1
llama-index-indices-managed-llama-cloud 0.6.3
llama-index-legacy 0.9.48.post4
llama-index-llms-openai 0.3.3
llama-index-multi-modal-llms-openai 0.3.0
llama-index-program-openai 0.3.1
llama-index-question-gen-openai 0.3.0
llama-index-readers-file 0.4.1
llama-index-readers-llama-parse 0.4.0
llama-parse 0.5.17
llvmlite 0.42.0
locket 1.0.0
logical-unification 0.4.6
loguru 0.7.3
lxml 5.3.0
lz4 4.3.3
Mako 1.3.8
mamba-ssm 2.2.4
MAPIE 0.9.1
marisa-trie 1.2.1
Markdown 3.7
markdown-it-py 3.0.0
MarkupSafe 3.0.2
marshmallow 3.23.1
matplotlib 3.7.5
matplotlib-inline 0.1.7
mdit-py-plugins 0.4.2
mdurl 0.1.2
memray 1.15.0
menuinst 2.1.2
mgarch 0.3.0
miniKanren 1.0.3
minorminer 0.2.15
mistune 3.0.2
ml-dtypes 0.4.1
mlflow 2.18.0
mlflow-skinny 2.18.0
mlforecast 0.13.4
mljar-scikit-plot 0.3.12
mljar-supervised 1.1.9
mlxtend 0.23.3
mmh3 2.5.1
model-index 0.1.11
modin 0.26.1
mplfinance 0.12.10b0
mpmath 1.3.0
msgpack 1.1.0
multidict 6.1.0
multipledispatch 1.0.0
multiprocess 0.70.16
multitasking 0.0.11
murmurhash 1.0.11
mypy-extensions 1.0.0
namex 0.0.8
narwhals 1.17.0
nbclient 0.10.1
nbconvert 7.16.4
nbformat 5.10.4
ndindex 1.9.2
nest-asyncio 1.6.0
networkx 3.4.2
neural-tangents 0.6.5
neuralprophet 0.9.0
nfoursid 1.0.1
ngboost 0.5.1
ninja 1.11.1.2
nixtla 0.6.4
nlpaug 1.1.11
nltk 3.8.1
nolds 0.6.1
nose 1.3.7
notebook 7.3.1
notebook_shim 0.2.4
numba 0.59.1
numba 0.59.1
numerapi 2.19.1
numexpr 2.10.2
numpy 1.26.4
nvidia-cublas-cu12 12.4.5.8
nvidia-cuda-cupti-cu12 12.4.127
nvidia-cuda-nvrtc-cu12 12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12 9.1.0.70
nvidia-cufft-cu12 11.2.1.3
nvidia-curand-cu12 10.3.5.147
nvidia-cusolver-cu12 11.6.1.9
nvidia-cusparse-cu12 12.3.1.170
nvidia-ml-py3 7.352.0
nvidia-nccl-cu12 2.21.5
nvidia-nvjitlink-cu12 12.4.127
nvidia-nvtx-cu12 12.4.127
nvtx 0.2.10
nx-cugraph-cu12 24.8.0
oauthlib 3.2.2
omegaconf 2.2.3
openai 1.57.0
opencensus 0.11.4
opencensus-context 0.1.3
opencv-contrib-python-headless 4.10.0.84
opencv-python 4.10.0.84
opendatalab 0.0.10
openmim 0.3.9
openpyxl 3.1.5
opentelemetry-api 1.28.2
opentelemetry-sdk 1.28.2
opentelemetry-semantic-conventions 0.49b2
openxlab 0.0.11
opt_einsum 3.4.0
optree 0.13.1
optuna 4.1.0
ordered-set 4.1.0
orjson 3.10.12
ortools 9.9.3963
osqp 0.6.7.post3
overrides 7.7.0
packaging 24.1
pandas 2.1.4
pandas-flavor 0.6.0
pandas_market_calendars 4.4.2
pandas_ta 0.3.14b0
pandocfilters 1.5.1
panel 1.5.4
param 2.1.1
parso 0.8.4
partd 1.4.2
pastel 0.2.1
pathos 0.3.2
patsy 1.0.1
pbr 6.1.0
pdf2image 1.17.0
peewee 3.17.3
peft 0.13.2
penaltymodel 1.1.0
PennyLane 0.39.0
PennyLane_Lightning 0.39.0
PennyLane-qiskit 0.36.0
persim 0.3.7
pexpect 4.9.0
pgmpy 0.1.26
pillow 10.4.0
pingouin 0.5.5
pip 24.0
platformdirs 3.10.0
plotly 5.24.1
plotly-resampler 0.10.0
plucky 0.4.3
pluggy 1.5.0
ply 3.11
pmdarima 2.0.4
polars 1.16.0
pomegranate 1.1.1
POT 0.9.5
pox 0.3.5
ppft 1.7.6.9
pprofile 2.2.0
preshed 3.0.9
p
r
e
s
h
e
d
3.0.9
p
r
o
m
e
t
h
e
u
s
_
c
l
i
e
n
t
0.2
1.1
p
r
o
m
p
t
_
t
o
o
l
k
i
t
3.0.4
8
p
r
o
p
c
a
c
h
e
0.2.1
p
r
o
p
h
e
t
1.1.6
p
r
o
t
o
-
p
l
u
s
1.2
5.0
p
r
o
t
o
b
u
f
5.2
9.1
p
s
u
t
i
l
5.9.8
p
t
y
p
r
o
c
e
s
s
0.7.0
P
u
L
P
2.9.0
p
u
r
e
_
e
v
a
l
0.2.3
p
y
-
c
p
u
i
n
f
o
9.0.0
p
y
-
h
e
a
t
0.0.6
p
y
-
h
e
a
t
-
m
a
g
i
c
0.0.2
p
y
_
l
e
t
s
_
b
e
_
r
a
t
i
o
n
a
l
1.0.1
p
y
-
s
p
y
0.4.0
p
y
_
v
o
l
l
i
b
1.0.1
p
y
4
j
0.1
0.9.7
p
y
a
m
l
2
4.9.0
p
y
a
r
r
o
w
1
6.1.0
p
y
a
s
n
1
0.6.1
p
y
a
s
n
1
_
m
o
d
u
l
e
s
0.4.1
p
y
b
i
n
d
1
1
2.1
3.6
p
y
c
a
r
e
t
3.3.2
p
y
c
o
s
a
t
0.6.6
p
y
c
p
a
r
s
e
r
2.2
1
p
y
c
r
y
p
t
o
d
o
m
e
3.2
1.0
p
y
c
t
0.5.0
p
y
d
a
n
t
i
c
2.9.2
p
y
d
a
n
t
i
c
_
c
o
r
e
2.2
3.4
P
y
D
M
D
2
0
2
4.1
2.1
p
y
e
r
f
a
2.0.1.5
p
y
f
o
l
i
o
-
r
e
l
o
a
d
e
d
0.9.8
P
y
g
m
e
n
t
s
2.1
8.0
P
y
J
W
T
2.1
0.1
p
y
k
a
l
m
a
n
0.9.7
p
y
l
e
v
1.4.0
p
y
l
i
b
c
u
g
r
a
p
h
-
c
u
1
2
2
4.8.0
p
y
l
i
b
r
a
f
t
-
c
u
1
2
2
4.8.1
p
y
l
u
a
c
h
2.2.0
p
y
m
a
n
n
k
e
n
d
a
l
l
1.4.3
p
y
m
c
5.1
9.0
p
y
m
d
p
t
o
o
l
b
o
x
4.0
b
3
p
y
n
n
d
e
s
c
e
n
t
0.5.1
3
p
y
n
v
j
i
t
l
i
n
k
-
c
u
1
2
0.4.0
p
y
n
v
m
l
1
1.4.1
p
y
o
d
2.0.2
p
y
o
g
r
i
o
0.1
0.0
P
y
o
m
o
6.8.2
p
y
p
a
r
s
i
n
g
3.2.0
p
y
p
d
f
5.1.0
p
y
p
o
r
t
f
o
l
i
o
o
p
t
1.5.6
p
y
p
r
o
j
3.7.0
P
y
Q
t
6
6.7.1
P
y
Q
t
6
-
Q
t
6
6.7.3
P
y
Q
t
6
_
s
i
p
1
3.9.0
p
y
r
b
1.0.1
p
y
r
e
-
e
x
t
e
n
s
i
o
n
s
0.0.3
2
p
y
r
o
-
a
p
i
0.1.2
p
y
r
o
-
p
p
l
1.9.1
p
y
s
i
m
d
j
s
o
n
6.0.2
P
y
S
o
c
k
s
1.7.1
p
y
s
p
n
e
g
o
0.1
1.2
p
y
s
t
a
n
3.1
0.0
p
y
t
e
n
s
o
r
2.2
6.4
p
y
t
e
s
s
e
r
a
c
t
0.3.1
0
p
y
t
e
s
t
8.3.4
p
y
t
e
s
t
-
r
u
n
n
e
r
6.0.1
p
y
t
h
o
n
-
d
a
t
e
u
t
i
l
2.9.0.p
o
s
t
0
p
y
t
h
o
n
-
j
s
o
n
-
l
o
g
g
e
r
2.0.7
p
y
t
h
o
n
-
s
t
a
t
e
m
a
c
h
i
n
e
2.5.0
p
y
t
o
r
c
h
-
f
o
r
e
c
a
s
t
i
n
g
1.2.0
p
y
t
o
r
c
h
-
i
g
n
i
t
e
0.5.1
p
y
t
o
r
c
h
-
l
i
g
h
t
n
i
n
g
2.4.0
p
y
t
o
r
c
h
-
m
e
t
r
i
c
-
l
e
a
r
n
i
n
g
2.3.0
p
y
t
o
r
c
h
-
t
a
b
n
e
t
4.1.0
p
y
t
z
2
0
2
4.2
p
y
v
i
n
e
c
o
p
u
l
i
b
0.6.5
p
y
v
i
z
_
c
o
m
m
s
3.0.3
P
y
W
a
v
e
l
e
t
s
1.7.0
P
y
Y
A
M
L
6.0.2
p
y
z
m
q
2
6.2.0
p
y
z
m
q
2
6.2.0
q
d
l
d
l
0.1.7.p
o
s
t
4
q
i
s
k
i
t
1.2.4
q
i
s
k
i
t
-
a
e
r
0.1
5.1
q
i
s
k
i
t
-
i
b
m
-
p
r
o
v
i
d
e
r
0.1
1.0
q
i
s
k
i
t
-
i
b
m
-
r
u
n
t
i
m
e
0.3
4.0
q
u
a
d
p
r
o
g
0.1.1
3
q
u
a
n
t
e
c
o
n
0.7.2
Q
u
a
n
t
L
i
b
1.3
6
Q
u
a
n
t
S
t
a
t
s
0.0.6
4
r
a
f
t
-
d
a
s
k
-
c
u
1
2
2
4.8.1
r
a
p
i
d
s
-
d
a
s
k
-
d
e
p
e
n
d
e
n
c
y
2
4.8.0
r
a
u
t
h
0.7.3
r
a
y
2.3
9.0
R
b
e
a
s
t
0.1.2
3
r
e
f
e
r
e
n
c
i
n
g
0.3
5.1
r
e
g
e
x
2
0
2
4.1
1.6
r
e
q
u
e
s
t
s
2.3
2.3
r
e
q
u
e
s
t
s
_
n
t
l
m
1.3.0
r
e
q
u
e
s
t
s
-
o
a
u
t
h
l
i
b
1.3.1
r
e
q
u
e
s
t
s
-
t
o
o
l
b
e
l
t
1.0.0
r
f
c
3
3
3
9
-
v
a
l
i
d
a
t
o
r
0.1.4
r
f
c
3
9
8
6
-
v
a
l
i
d
a
t
o
r
0.1.1
r
i
c
h
1
3.9.4
r
i
p
s
e
r
0.6.1
0
R
i
s
k
f
o
l
i
o
-
L
i
b
6.1.1
r
i
s
k
p
a
r
i
t
y
p
o
r
t
f
o
l
i
o
0.6.0
r
i
v
e
r
0.2
1.0
r
m
m
-
c
u
1
2
2
4.8.2
r
p
d
s
-
p
y
0.2
2.3
r
s
a
4.9
r
u
a
m
e
l.y
a
m
l
0.1
8.6
r
u
a
m
e
l.y
a
m
l.c
l
i
b
0.2.8
r
u
p
t
u
r
e
s
1.1.9
r
u
s
t
w
o
r
k
x
0.1
5.1
s
3
t
r
a
n
s
f
e
r
0.1
0.4
s
a
f
e
t
e
n
s
o
r
s
0.4.5
S
A
L
i
b
1.5.1
s
c
h
e
m
d
r
a
w
0.1
5
s
c
i
k
e
r
a
s
0.1
3.0
s
c
i
k
i
t
-
b
a
s
e
0.7.8
s
c
i
k
i
t
-
i
m
a
g
e
0.2
2.0
s
c
i
k
i
t
-
l
e
a
r
n
1.4.2
s
c
i
k
i
t
-
l
e
a
r
n
-
e
x
t
r
a
0.3.0
s
c
i
k
i
t
-
o
p
t
i
m
i
z
e
0.1
0.2
s
c
i
k
i
t
-
p
l
o
t
0.3.7
s
c
i
k
i
t
-
t
d
a
1.1.1
s
c
i
p
y
1.1
1.4
s
c
s
3.2.7
s
d
e
i
n
t
0.3.0
s
e
a
b
o
r
n
0.1
3.2
S
e
n
d
2
T
r
a
s
h
1.8.3
s
e
n
t
e
n
c
e
-
t
r
a
n
s
f
o
r
m
e
r
s
3.3.1
s
e
n
t
e
n
c
e
p
i
e
c
e
0.2.0
s
e
q
e
v
a
l
1.2.2
s
e
t
u
p
t
o
o
l
s
6
5.5.0
s
e
t
u
p
t
o
o
l
s
-
s
c
m
8.1.0
s
h
a
p
0.4
6.0
s
h
a
p
e
l
y
2.0.6
S
h
i
m
m
y
2.0.0
s
i
m
p
e
r
v
i
s
o
r
1.0.0
s
i
m
p
l
e
j
s
o
n
3.1
9.3
s
i
m
p
y
4.1.1
s
i
x
1.1
7.0
s
k
l
e
a
r
n
-
j
s
o
n
0.1.0
s
k
t
i
m
e
0.2
6.0
s
l
i
c
e
r
0.0.8
s
m
a
r
t
-
o
p
e
n
7.0.5
s
m
m
a
p
5.0.1
s
n
i
f
f
i
o
1.3.1
s
o
r
t
e
d
c
o
n
t
a
i
n
e
r
s
2.4.0
s
o
u
p
s
i
e
v
e
2.6
s
p
a
c
y
3.7.5
s
p
a
c
y
-
l
e
g
a
c
y
3.0.1
2
s
p
a
c
y
-
l
o
g
g
e
r
s
1.0.5
S
Q
L
A
l
c
h
e
m
y
2.0.3
6
s
q
l
p
a
r
s
e
0.5.3
s
r
s
l
y
2.5.0
s
s
m
0.0.1
s
t
a
b
l
e
_
b
a
s
e
l
i
n
e
s
3
2.4.0
s
t
a
c
k
-
d
a
t
a
0.6.3
s
t
a
n
i
o
0.5.1
stanio 0.5.1
statsforecast 1.7.8
statsmodels 0.14.4
stevedore 5.4.0
stochastic 0.6.0
stockstats 0.6.2
stopit 1.1.2
striprtf 0.0.26
stumpy 1.13.0
symengine 0.13.0
sympy 1.13.1
ta 0.11.0
ta-lib 0.5.1
tables 3.10.1
tabulate 0.8.10
tadasets 0.2.1
tbats 1.1.3
tblib 3.0.0
tenacity 8.5.0
tensorboard 2.18.0
tensorboard-data-server 0.7.2
tensorboardX 2.6.2.2
tensorflow 2.18.0
tensorflow-addons 0.23.0
tensorflow_decision_forests 1.11.0
tensorflow-io-gcs-filesystem 0.37.1
tensorflow-probability 0.25.0
tensorflow-text 2.18.0
tensorly 0.9.0
tensorrt 10.7.0
tensorrt_cu12 10.7.0
tensorrt-cu12-bindings 10.7.0
tensorrt-cu12-libs 10.7.0
tensortrade 1.0.3
termcolor 2.5.0
terminado 0.18.1
text-unidecode 1.3
textual 1.0.0
tf_keras 2.18.0
tf2jax 0.3.6
thinc 8.2.5
threadpoolctl 3.5.0
thundergbm 0.3.17
tifffile 2024.9.20
tigramite 5.2.6.7
tiktoken 0.8.0
timm 1.0.3
tinycss2 1.4.0
tinygrad 0.10.0
tokenizers 0.20.3
toml 0.10.2
toolz 0.12.1
torch 2.5.1
torch_cluster 1.6.3
torch-geometric 2.6.1
torch_scatter 2.1.2
torch_sparse 0.6.18
torch_spline_conv 1.2.2
torchdata 0.10.0
torchmetrics 1.2.1
torchvision 0.20.1
tornado 6.4.2
TPOT 0.12.2
tqdm 4.66.5
traitlets 5.14.3
transformers 4.46.3
treelite 4.3.0
triad 0.9.8
triton 3.1.0
truststore 0.8.0
tsdownsample 0.1.3
tsfel 0.1.9
tsfresh 0.20.2
tslearn 0.6.3
tweepy 4.14.0
typeguard 2.13.3
typer 0.9.4
typer-config 1.4.2
types-python-dateutil 2.9.0.20241206
typing_extensions 4.12.2
typing-inspect 0.9.0
Request New Libraries
To request a new library, contact us . We will add the library to the queue for review and deployment. Since the libraries run on
our servers, we need to ensure they are secure and won't cause harm. The process of adding new libraries takes 2-4 weeks to
complete. View the list of libraries currently under review on the Issues list of the Lean GitHub repository .
tzdata 2024.2
uc-micro-py 1.0.3
ucx-py-cu12 0.39.2
ucxx-cu12 0.39.1
umap-learn 0.5.7
update-checker 0.18.0
uri-template 1.3.0
uritemplate 4.1.1
urllib3 2.2.3
utilsforecast 0.2.4
virtualenv 20.28.0
wasabi 1.1.3
wcwidth 0.2.13
weasel 0.4.1
webargs 8.6.0
webcolors 24.11.1
webencodings 0.5.1
websocket-client 1.8.0
websockets 14.1
Werkzeug 3.1.3
wheel 0.44.0
widgetsnbextension 4.0.13
window_ops 0.0.15
woodwork 0.31.0
wordcloud 1.9.4
wrapt 1.16.0
wurlitzer 3.1.1
x-transformers 1.42.24
xarray 2024.11.0
xarray-einstats 0.8.0
xgboost 2.1.3
xlrd 2.0.1
XlsxWriter 3.2.0
xxhash 3.5.0
xyzservices 2024.9.0
yarl 1.18.3
ydf 0.9.0
yellowbrick 1.5
yfinance 0.2.50
zict 3.0.0
zipp 3.21.0
zope.event 5.0
zope.interface 7.2
zstandard 0.23.0
Projects > LEAN Engine Versions
Projects
LEAN Engine Versions
Introduction
The latest master branch on the LEAN GitHub repository is the default engine branch that runs the backtesting, research, and
live trading nodes in QuantConnect Cloud. The latest version of LEAN is generally the safest as it includes all bug fixes.
Trading Firm or Institution tier users concerned for stability can elect to use older or custom versions of LEAN in the IDE. These
are powered by the QuantConnect/LEAN Github Branches . We use a continuous deployment process to ship custom branches
to production for trading. To create a custom version of LEAN, make a pull request to LEAN which will be reviewed by our team.
Change Branches
Follow these steps to change the LEAN engine branch that runs your backtests and live trading algorithms:
1. Open a project .
2. In the Project panel, click the LEAN Engine field and then click a branch from the drop-down menu.
3. (Optional) Click About Version to display the branch description.
4. If you want to always use the master branch, select the Always use Master Branch check box.
5. Click Select .
Changing the Lean engine branch only affects the current project. If you create a new project , the new project will use the
master branch by default.
Request New Branches
Before starting a pull-request to create a new branch, contact us to discuss the goals of the project.
Research
Research
The Research Environment is a Jupyter notebook -based environment where you can access our data through the QuantBook
class instead of through the QCAlgorithm class in a backtest. The environment supports both Python and C#. If you use Python,
you can import code from the code files in your project into the Research Environment to aid development.
Getting Started
Learn the basics
Deployment
Spin up a notebook
See Also
Research Environment Product
Charting
Object Store
Research > Getting Started
Research
Getting Started
Introduction
The Research Environment is a Jupyter notebook -based, interactive commandline environment where you can access our data
through the QuantBook class. The environment supports both Python and C#. If you use Python, you can import code from the
code files in your project into the Research Environment to aid development.
Before you run backtests, we recommend testing your hypothesis in the Research Environment. It's easier to perform data
analysis and produce plots in the Research Environment than in a backtest. The Research Environment also supports debugging
for Python projects.
Before backtesting or live trading with machine learning models, you may find it beneficial to train them in the Research
Environment, save them in the Object Store, and then load them from the Object Store into the backtesting and live trading
environment
In the Research Environment, you can also use the QuantConnect API to import your backtest results for further analysis.
Note: This chapter is an introduction to the Research Environment for the Algorithm Lab. For more comprehensive information
on using research notebooks, see our dedicated Research Environment documentation.
Example
The following snippet demonstrates how to use the Research Environment to plot the price and Bollinger Bands of the S&P 500
index ETF, SPY:
Open Notebooks
Each new project you create contains a notebook file by default. Follow these steps to open the notebook:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, expand the Workspace (Workspace) section.
4. Click the research.ipynb file.
When you open a notebook, it automatically tries to connect to the correct Jupyter server and select the correct kernel, which
can take up to one minute. If the top-right corner of the notebook displays a base (Python x.x.x) button, wait for the button to
change to Foundation-Py-Default before you run the cells. If you run cells before the notebook connects to the server and
kernel, you may get the following error message:
NameError: name 'QuantBook' is not defined
Run Notebook Cells
# Create a QuantBook
qb = QuantBook()
# Add an asset.
symbol = qb.add_equity("SPY").symbol
# Request some historical data.
history = qb.history(symbol, 360, Resolution.DAILY)
# Calculate the Bollinger Bands.
bbdf = qb.indicator(BollingerBands(30, 2), symbol, 360, Resolution.DAILY)
# Plot the data
bbdf[['price', 'lowerband', 'middleband', 'upperband']].plot();
PY
Notebooks are a collection of cells where you can write code snippets or MarkDown. To execute a cell, press Shift+Enter .
The following describes some helpful keyboard shortcuts to speed up your research:
Keyboard Shortcut Description
Shift+Enter Run the selected cell.
a Insert a cell above the selected cell.
b Insert a cell below the selected cell.
x Cut the selected cell.
v Paste the copied or cut cell.
z Undo cell actions.
There is a 15 minute timeout period before the cells becomes unresponsive. If this occurs, restart the notebook to be able to run
cells again.
Stop Nodes
You need stop node permissions to stop research nodes in the cloud.
Follow these steps to stop a research node:
1. Open the project .
2. In the right navigation menu, click the Resources icon.
3. Click the stop button next to the research node you want to stop.
Add Notebooks
Follow these steps to add notebook files to a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, expand the Workspace (Workspace) section.
4. Click the New File icon.
5. Enter fileName .ipynb .
6. Press Enter .
Rename Notebooks
Follow these steps to rename a notebook in a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, right-click the notebook you want to rename and then click Rename .
4. Enter the new name and then press Enter .
The following directory names are reserved: .ipynb_checkpoints , .idea , .vscode , __pycache__ , bin , obj , backtests , live ,
optimizations , storage , and report .
Delete Notebooks
Follow these steps to delete a notebook in a project:
1. Open the project .
2. In the right navigation menu, click the Explorer icon.
3. In the Explorer panel, right-click the notebook you want to delete and then click Delete Permanently .
4. Click Delete .
Learn Jupyter
The following table lists some helpful resources to learn Jupyter:
Type Name Producer
Text Jupyter Tutorial tutorialspoint
Text
Jupyter Notebook Tutorial: The
Definitive Guide
DataCamp
Text An Introduction to DataFrame Microsoft Developer Blogs
Research > Deployment
Research
Deployment
Introduction
This page is an introduction to the Research Environment for the Algorithm Lab. For more comprehensive information on using
research notebooks, see the Research Environment documentation product.
Resources
Research nodes enable you to spin up an interactive, command-line, Jupyter Research Environment . Several models of
research nodes are available. More powerful research nodes allow you to handle more data and run faster computations in your
notebooks. The following table shows the specifications of the research node models:
Name Number of Cores
Processing Speed
(GHz)
RAM (GB) GPU
R1-4 1 2.4 4 0
R2-8 2 2.4 8 0
R4-12 4 2.4 12 0
R4-16-GPU 4 3 16 1/3
R8-16 8 2.4 16 0
Refer to the Pricing page to see the price of each research node model. You get one free R1-4 research node in your first
organization, but the node is replaced when you subscribe to a paid research node in the organization.
To view the status of all of your organization's nodes, see the Resources panel of the IDE. When you launch the Research
Environment, it uses the best-performing resource by default, but you can select a specific resource to use .
The CPU nodes are available on a fair usage basis. The GPU nodes can be shared with a maximum of three members. Depending
on the server load, you may use all of the GPU's processing power.
Sharing
The Research Environment does not currently support simultaneous coding between two peers. You can view the notebook of
your colleague as they edit it, but you can't both edit the notebook at the same time. To enable your team members to see your
notebook, add them to the project .
Sharing your notebooks is helpful when publishing your findings online, getting feedback on your research process, and
explaining your methodology to others. To share a notebook with members not in your organization, run a backtest , generate a
link to share the backtest , and then give the backtest link to the other person. They will be able to clone the project and launch
the Research Environment using the notebook files.
Historical Data
On our platform, you can access historical data for all of the asset classes that we support. We have datasets for Equities,
Options, Futures, Crypto, Forex, CFD, Indices, and alternative data. View the Dataset Market to see all of the datasets that we
support. To import custom data, see Custom Data .
Look-Ahead Bias
Look-ahead bias occurs when an algorithm makes decisions using data that would not have yet been available. As you work in
the Research Environment, take measures to reduce look-ahead bias in your workflow. If look-ahead bias seeps into your
research, you may find profitable results on historical data, however, you will not be able to apply your findings in a live trading
algorithm because the data does not exist in real-time.
An example of look-ahead bias is using the closing price to make trading decisions and filling your orders at the same closing
price. This can happen in the Research Environment because all of the data is available at the same time. This type of lookahead bias cannot occur when backtesting with Lean because the data is fed into your algorithm in discrete slices of time and an
order is filled using a slice that occurs after the order was placed.
A second example of look-ahead bias occurs when researchers test a model on the same dataset that was used to train the
model. In this situation, the model is trained using data that would not have been available when the model was created in
practice. A third example of look-ahead bias is when researchers select a universe to apply their trading strategy to based on
the past performance of the universe.
Backtesting
Backtesting
Backtesting is the process of simulating a trading algorithm on historical data. By running a backtest, you can measure how the
algorithm would have performed in the past. Although past performance doesn't guarantee future results, an algorithm that has
a proven track record can provide investors with more confidence when deploying to live trading than an algorithm that hasn't
performed favorably in the past. Use the QuantConnect platform to run your backtests because we have institutional-grade
datasets, an open-source backtesting engine that's constantly being improved, cloud servers to execute the backtests, and the
backtesting hardware is maintained 24/7 by QuantConnect engineers.
Getting Started
Learn the basics
Research Guide
Backtesting 101
Deployment
Cloud backtests with institutional-grade data
Results
Gathering historical results
Debugging
Solve coding errors
Report
Measuring algorithm performance
Engine Performance
Track LEAN performance metrics
See Also
Writing Algorithms
Optimization
Resources
Backtesting > Getting Started
Backtesting
Getting Started
Introduction
Backtesting is the process of simulating a trading algorithm on historical data. By running a backtest, you can measure how the
algorithm would have performed in the past. Although past performance doesn't guarantee future results, an algorithm that has
a proven track record can provide investors with more confidence when deploying to live trading than an algorithm that hasn't
performed favorably in the past. Use the QuantConnect platform to run your backtests because we have institutional-grade
datasets, an open-source backtesting engine that's constantly being improved, cloud servers to execute the backtests, and the
backtesting hardware is maintained 24/7 by QuantConnect engineers.
Run Backtests
To run a backtest, open a project and then click the Backtest icon. If the project successfully builds, "Received backtest
backtestName request" displays. If the backtest successfully launches, the IDE displays the backtest results page in a new tab.
If the backtest fails to launch due to coding errors, the new tab displays the error. As the backtest executes, you can refresh or
close the IDE without interfering with the backtest because it runs on our cloud servers. The backtest runs up to the start of the
out-of-sample hold out period .
View All Backtests
Follow these steps to view all of the backtests of a project:
1. Open the project that contains the backtests you want to view.
2. In the top-right corner of the IDE, click the Backtest Results icon.
A table containing all of the backtest results for the project is displayed. If there is a play icon to the left of the name, it's a
backtest result . If there is a fast-forward icon next to the name, it's an optimization result .
3. (Optional) In the top-right corner, select the Show field and then select one of the options from the drop-down menu to
filter the table by backtest or optimization results.
4. (Optional) In the bottom-right corner, click the Hide Error check box to remove backtest and optimization results from the
table that had a runtime error.
5. (Optional) Use the pagination tools at the bottom to change the page.
6. (Optional) Click a column name to sort the table by that column.
7. Click a row in the table to open the results page of that backtest or optimization.
Rename Backtests
We give an arbitrary name (for example, "Smooth Apricot Chicken") to your backtest result files, but you can follow these steps
to rename them:
1. Open the backtest history of the project.
2. Hover over the backtest you want to rename and then click the pencil icon that appears.
3. Enter the new backtest name and then click OK .
To programmatically set the backtest name, call the set_name method.
For more information, see Set Name and Tags .
Out of Sample Period
To reduce the chance of overfitting, organization managers can enforce all backtests must end a certain number of months
before the current date. For example, if you set a one year out-of-sample period, the researchers on your team will not be able
to use the most recent year of data in their backtests. A out-of-sample period is helpful because it leaves you a period to test
your model after your done the development stage. Follow these steps to change the backtest out-of-sample period:
self.set_name("Backtest Name")
PY
1. Open the organization homepage .
2. Scroll down to the Backtesting Out of Sample Period section.
3. Adjust the out-of-sample period duration or click on "No Holdout Period".
On-Premise Backtests
For information about on-premise backtests with Local Platform , see Getting Started .
Get Backtest Id
To get the backtest Id, see the first line of the log file . An example backtest Id is 8b16cec0c44f75188d82f9eadb310e17.
Share Backtests
The backtest results page enables you to share your backtest results. You need to make a backtest public in order to share it. To
make a backtest public, in the Share Results section of the backtest results page, click Make Public . Once you make a backtest
public, click Backtest URL to copy a link to the backtest result or click Embed Code to copy an iframe HTML element you can
embed on a website.
The following widget is an example of an embedded backtest result:
CHARTS
CODE
CLONE
STRATEGY REPORT
MEASURED APRICOT CHICKEN
To attach the embedded backtest result to a forum discussion, see Create Discussions or Post Comments .
After you've made your backtest results public, the results are always stored and anyone with the link can access the results. To
make your backtest results private again, click Make Private .
Backtesting > Research Guide
Backtesting
Research Guide
Introduction
QuantConnect aims to teach and inspire our community to create high-performing algorithmic trading strategies. We measure
our success by the profits created by the community through their live trading. As such, we try to build the best quantitative
research techniques possible into the product to encourage a robust research process.
Hypothesis-Driven Research
We recommend you develop an algorithmic trading strategy based on a central hypothesis. You should develop an algorithm
hypothesis at the start of your research and spend the remaining time exploring how to test your theory. If you find yourself
deviating from your core theory or introducing code that isn't based around that hypothesis, you should stop and go back to
thesis development.
Wang et al. (2014) illustrate the danger of creating your hypothesis based on test results. In their research, they examined the
earnings yield factor in the technology sector over time. During 1998-1999, before the tech bubble burst, the factor was
unprofitable. If you saw the results and then decided to bet against the factor during 2000-2002, you would have lost a lot of
money because the factor performed extremely well during that time.
Hypothesis development is somewhat of an art and requires creativity and great observation skills. It is one of the most powerful
skills a quant can learn. We recommend that an algorithm hypothesis follow the pattern of cause and effect. Your aim should be
to express your strategy in the following sentence:
A change in {cause} leads to an {effect}.
To search for inspiration, consider causes from your own experience, intuition, or the media. Generally, causes of financial
market movements fall into the following categories:
Human psychology
Real-world events/fundamentals
Invisible financial actions
Consider the following examples:
Cause leads to Effect
Share class stocks are the same company, so any
price divergence is irrational...
A perfect pairs trade. Since they are the same
company, the price will revert.
New stock addition to the S&P500 Index causes fund
managers to buy up stock...
An increase in the price of the new asset in the
universe from buying pressure.
Increase in sunshine-hours increases the production
of oranges...
An increase in the supply of oranges, decreasing
the price of Orange Juice Futures.
Allegations of fraud by the CEO causes investor faith
in the stock to fall...
A collapse of stock prices for the company as
people panic.
FDA approval of a new drug opens up new markets
for the pharmaceutical company... A jump in stock prices for the company.
Increasing federal interest rates restrict lending from
banks, raising interest rates...
Restricted REIT leverage and lower REIT ETF
returns.
There are millions of potential alpha strategies to explore, each of them a candidate for an algorithm. Once you have chosen a
strategy, we recommend exploring it for no more than 8-32 hours, depending on your coding ability.
Research Panel
We launched the Research Guide in 2019 to inform you about common quantitative
research pitfalls. It displays a power gauge for the number of backtests performed, the
number of parameters used, and the time invested in the strategy. These measures can
give a ballpark estimate of the overfitting risk of the project. Generally, as a strategy
becomes more overfit on historical data, it is less likely to perform well in live trading.
These properties were selected based on the recommended best practices of the global
quantitative research community.
Restricting Backtests
According to current research, the number of backtests performed on an idea should be limited to prevent overfitting. In
theory, each backtest performed on an idea moves it one step closer to being overfitted as you are testing and selecting
for strategies written into your code instead of being based on a central thesis. For more information, see the paper Probability
of Backtest Overfitting (Bailey, Borwein, Jim Zho, & López de Prado, 2015).
QuantConnect does not restrict the number of backtests performed on a project, but we have implemented the counter as a
guide for your reference. Your coding skills are a factor in how many backtests constitute overfitting, so if you are a new
programmer you can increase these targets.
Backtest Count Overfit Reference
0-30: Likely Not Overfit 30-70: Possibly Overfitting 70+ Probably Overfitting
Reducing Strategy Parameters
With just a handful of parameters, it is possible to create an algorithm that perfectly models historical markets. Current
research suggests keeping your parameter count to a minimum to decrease the risk of overfitting.
Parameter Overfit Reference
0-10: Likely Not Overfit 10-20: Possibly Overfitting 20+ Probably Overfitting
Limiting Research Time Invested
As you spend more time on one algorithm, research suggests you are more likely to be overfitting the strategy to the
data. It is common to become attached to an idea and spend weeks or months to perform well in a backtest. Assuming
you are a proficient coder who fully understands the QuantConnect API, we recommend no more than 16 hours of work per
experiment. In theory, within two full working days, you should be able to test a single hypothesis thoroughly.
Research Time Overfitting Reference
0-8 Hours: Likely Not Overfit 8-16 Hours: Possibly Overfitting 16 Hours+ Probably Overfitting
Parameter Detection
Using parameters is almost unavoidable, but a strategy trends toward being overfitted as more parameters get added or finetuned. Adding or optimizing parameters should only be done by a robust methodology such as walk-forward optimization . The
parameter detection system is a general guide to inform you of how many parameters are present in the algorithm. It looks for
criteria to warn that code is potentially a parameter. The following table shows the criteria for parameters:
Parameter Types Example Instances
Numeric Comparison Numeric operators used to compare numeric arguments: <= < > >=
Time Span Setting the interval of TimeSpan or timedelta
Order Event Inputting numeric arguments when placing orders
Scheduled Event Inputting numeric arguments when scheduling an algorithm event to occur
Variable Assignment Assigning numeric values to variables
Mathematical Operation Any mathematical operation involving explicit numbers
Lean API Numeric arguments passed to Indicators, Consolidators, Rolling Windows, etc.
The following table shows common expressions that are not parameters:
Non-Parameter Types Example Instances
Common APIs set_start_date , set_end_date , set_cash , etc.
Boolean Comparison Testing for True or False conditions
String Numbers Numbers formatted as part of log method or debug method statements
Variable Names Any variable names that use numbers as part of the name (for example, smaIndicator200 )
Common Functions Rounding, array indexing, boolean comparison using 1/0 for True/False, etc.
Overfitting
Overfitting occurs when you fine-tune the parameters of an algorithm to fit the detail and noise of backtesting data to the extent
that it negatively impacts the performance of the algorithm on new data. The problem is that the parameters don't necessarily
apply to new data and thus negatively impact the algorithm's ability to generalize and trade well in all market conditions. The
following table shows ways that overfitting can manifest itself:
Data Practice Description
Data Dredging
Performing many statistical tests on data and only paying attention to those that come back
with significant results.
Hyper-Tuning Parameters
Manually changing algorithm parameters to produce better results without altering the test
data.
Overfit Regression Models
Regression, machine learning, or other statistical models with too many variables will likely
introduce overfitting to an algorithm.
Stale Testing Data
Not changing the backtesting data set when testing the algorithm. Any improvements might
not be able to be generalized to different datasets.
An algorithm that is dynamic and generalizes to new data is more valuable to funds and individual investors. It is more likely to
survive across different market conditions and apply to new asset classes and markets.
Out of Sample Period
To reduce the chance of overfitting, organization managers can enforce all backtests must end a certain number of months
before the current date. For example, if you set a one year out-of-sample period, the researchers on your team will not be able
to use the most recent year of data in their backtests. A out-of-sample period is helpful because it leaves you a period to test
your model after your done the development stage. Follow these steps to change the backtest out-of-sample period:
1. Open the organization homepage .
2. Scroll down to the Backtesting Out of Sample Period section.
3. Adjust the out-of-sample period duration or click on "No Holdout Period".
Backtesting > Deployment
Backtesting
Deployment
Introduction
Deploy a backtest to simulate your trading algorithm on our cloud servers. Since the same Lean engine is used to run backtests
and live trading algorithms, it's easy to transition from backtesting to live trading once you are satisfied with the historical
performance of your algorithm. If you find any issues with Lean, the historical data, or the website when running backtests, we'll
resolve the issue.
Nodes
Backtesting nodes enable you to run backtests. The more backtesting nodes your organization has, the more concurrent
backtests that you can run. Several models of backtesting nodes are available. Backtesting nodes that are more powerful can
run faster backtests and backtest nodes with more RAM can handle more memory-intensive operations like training machine
learning models, processing Options data, and managing large universes. The following table shows the specifications of the
backtesting node models:
Name Number of Cores
Processing Speed
(GHz)
RAM (GB) GPU
B-MICRO 2 3.3 8 0
B2-8 2 4.9 8 0
B4-12 4 4.9 12 0
B4-16-GPU 4 3 16 1/3
B8-16 8 4.9 16 0
Refer to the Pricing page to see the price of each backtesting node model. You get one free B-MICRO backtesting node in your
first organization. This node incurs a 20-second delay when you launch backtests, but the delay is removed and the node is
replaced when upgrade your organization to a paid tier and add a new backtesting node .
To view the status of all of your organization's nodes, see the Resources panel of the IDE. When you run a backtest, it uses the
best-performing resource by default, but you can select a specific resource to use .
The CPU nodes are available on a fair usage basis while the GPU nodes can be shared with a maximum of three members.
Depending on the server load, you may use all of the GPU's processing power. GPU nodes perform best on repetitive and
highly-parallel tasks like training machine learning models. It takes time to transfer the data to the GPU for computation, so if
your algorithm doesn't train machine learning models, the extra time it takes to transfer the data can make it appear that GPU
nodes run slower than CPU nodes.
Concurrent Backtesting
Concurrent backtesting is the process of running multiple backtests at the same time within a single organization. Concurrent
backtesting speeds up your strategy development because you don't have to wait while a single backtest finishes executing.
You need multiple backtesting nodes in your organization to run concurrent backtests. The more backtesting nodes that your
organization has, the more concurrent backtests you can execute. If you are trying to fine-tune your parameters , consider
running a parameter optimization . If you run a parameter optimization job, you can run multiple backtests concurrently without
having multiple backtest nodes.
The number of backtesting nodes that you can have in your organization depends on the tier of your organization. The following
table shows the backtesting node quotas:
Tier Node Quota
Free 1
Quant Researcher 2
Team 10
Trading Firm Unlimited
Institution Unlimited
Logs
The amount of logs you can generate per backtest and per day depends on the tier of your organization. The following table
shows the amount of logs you can generate on each tier:
Tier Logs Per Backtest Logs Per Day
Free 10KB 3MB
Quant Researcher 100KB 3MB
Team 1MB 10MB
Trading Firm 5MB 50MB
Institution Inf. Inf.
To check the log storage space you have remaining, log in to the Algorithm Lab and then, in the left navigation bar, click
Organization > Resources .
Orders
The number of orders you can place in a single backtest depends on the tier of your organization. The following table shows the
number of orders you can place on each tier:
Tier Orders Quota
Free 10K
Quant Researcher 10M
Team Unlimited
Trading Firm Unlimited
Institution Unlimited
Security
Your code is stored in a database, isolated from the internet. When the code leaves the database, it is compiled and obfuscated
before being deployed to the cloud. If the cloud servers were compromised, this process makes it difficult to read your strategy.
As we've seen over recent years, there can never be any guarantee of security with online websites. However, we deploy all
modern and common security procedures. We deploy nightly software updates to keep the server up to date with the latest
security patches. We also use SSH key login to avoid reliance on passwords. Internally, we use processes to ensure only a
handful of people have access to the database and we always restrict logins to never use root credentials.
See our Security and IP documentation for more information.
Build Projects
If the compiler finds errors during the build process, the IDE highlights the lines of code that caused the errors in red. Your
projects will automatically build after each keystroke. To manually build a project, open the project and then click the Build
icon. If the build process is unresponsive, try refreshing the page and building again. If the build process continues to be
unresponsive, check our Status page or contact us .
Run Backtests
To run a backtest, open a project and then click the Backtest icon. If the project successfully builds, "Received backtest
backtestName request" displays. If the backtest successfully launches, the IDE displays the backtest results page in a new tab.
If the backtest fails to launch due to coding errors, the new tab displays the error. As the backtest executes, you can refresh or
close the IDE without interfering with the backtest because it runs on our cloud servers. The backtest runs up to the start of the
out-of-sample hold out period .
Stop Backtests
To stop a running backtest, open the Resources panel , and then click the stop icon next to the backtest node. You can stop
nodes that you are using, but you need stop node permissions to stop nodes other members are using.
Runtime Quota
The maximum amount of time a backtest can run for is 12 hours. The runtime depends on the amount of data, the algorithm
features and computations, and the backtest node type.
Backtesting > Results
Backtesting
Results
Introduction
The backtest results page shows your algorithm's performance. Review the results page to see how your algorithm has
performed during the backtest and to investigate how you might improve your algorithm before live trading.
View Backtest Results
The backtest results page automatically displays when you deploy a backtest . The backtest results page presents the equity
curve, trades, logs, performance statistics, and much more information.
The content in the backtest results page updates as your backtest executes. You can close or refresh the window without
interrupting the backtest because the backtesting node processes on our servers. If you close the page, to open the page again,
view all of the project's backtests . Unless you explicitly make the backtest public, only you can view its results. If you delete a
backtest result or you are inactive for 12 months, we archive your backtest results.
Runtime Statistics
The banner at the top of the backtest results page displays the runtime statistics of your backtest.
The following table describes the default runtime statistics:
Statistic Description
Equity The total portfolio value if all of the holdings were sold at current market rates.
Fees The total quantity of fees paid for all the transactions.
Holdings The absolute sum of the items in the portfolio.
Net Profit The dollar-value return across the entire trading period.
PSR
The probability that the estimated Sharpe ratio of an algorithm is greater than a benchmark
(1).
Return The rate of return across the entire trading period.
Unrealized
The amount of profit a portfolio would capture if it liquidated all open positions and paid the
fees for transacting and crossing the spread.
Volume The total value of assets traded for all of an algorithm's transactions.
To download the runtime statistics data, see Download Results .
To add a custom runtime statistic, see Add Statistics .
Built-in Charts
The backtest results page displays a set of built-in charts to help you analyze the performance of your algorithm. The following
table describes the charts displayed on the page:
Chart Description
Strategy Equity A time series of equity and periodic returns.
Capacity A time series of strategy capacity snapshots.
Drawdown A time series of equity peak-to-trough value.
Benchmark A time series of the benchmark closing price (SPY, by default).
Exposure A time series of long and short exposure ratios.
Assets Sales Volume A chart showing the proportion of total volume for each traded security.
Portfolio Turnover A time series of the portfolio turnover rate.
Portfolio Margin
A stacked area chart of the portfolio margin usage. For more information about this chart,
see Portfolio Margin Plots .
Asset Plot A time series of an asset's price with order event annotations. These charts are available for
all paid organziation tiers. For more information about these charts, see Asset Plots .
To download the chart data, see Download Results .
Asset Plots
Asset plots display the trade prices of an asset and the following order events you have for the asset:
Order Event Icon
Submissions Gray circle
Updates Blue circle
Cancellations Gray square
Fills and partial fills Green (buys) or red (sells) arrows
The following image shows an example asset plot for AAPL:
The order submission icons aren't visible by default.
View Plots
Follow these steps to open an asset plot:
1. Open the backtest results page.
2. Click the Orders tab.
3. Click the Asset Plot icon that's next to the asset Symbol in the Orders table.
Tool Tips
When you hover over one of the order events in the table, the asset plot highlights the order event, displays the asset price at
the time of the event, and displays the tag associated with the event. Consider adding helpful tags to each order event to help
with debugging your algorithm. For example, when you cancel an order, you can add a tag that explains the reason for
cancelling it.
Adjust the Display Period
The resolution of the asset price time series in the plot doesn't necessarily match the resolution you set when you subscribed to
the asset in your algorithm. If you are displaying the entire price series, the series usually displays the daily closing price.
However, when you zoom in, the chart will adjust its display period and may use higher resolution data. To zoom in and out,
perform either of the following actions:
Click the 1m , 3m , 1y , or All period in the top-right corner of the chart.
Click a point on the chart and drag your mouse horizontally to highlight a specific period of time in the chart.
If you have multiple order events in a single day and you zoom out on the chart so that it displays the daily closing prices, the
plot aggregates the order event icons together as the price on that day.
Order Fill Prices
The plot displays fill order events at the actual fill price of your orders. The fill price is usually not equal to the asset price that
displays because of the following reasons:
Your order experiences slippage .
If you use quote data, your order fills at the bid or ask price.
The fill model may fill your order at the high or low price.
Custom Charts
The results page shows the custom charts that you create.
Supported Chart Types
We support the following types of charts:
If you use SeriesType.Candle and plot enough values, the plot displays candlesticks. However, the plot method only accepts
one numerical value per time step, so you can't plot candles that represent the open, high, low, and close values of each bar in
your algorithm. The charting software automatically groups the data points you provide to create the candlesticks, so you can't
control the period of time that each candlestick represents.
To create other types of charts, save the plot data in the Object Store and then load it into the Research Environment. In the
Research Environment, you can create other types of charts with third-party charting packages .
Supported Markers
When you create scatter plots, you can set a marker symbol. We support the following marker symbols:
Chart Quotas
Intensive charting requires hundreds of megabytes of data, which is too much to stream online or display in a web browser. The
number of series and the number of data points per series you can plot depends on your organization tier . The following table
shows the quotas:
Tier Max Series Max Data Points per Series
Free 10 4,000
Quant Researcher 10 8,000
Team 25 16,000
Trading Firm 25 32,000
Institution 100 96,000
If you exceed the series quota, your algorithm stops executing and the following message displays:
Exceeded maximum chart series count, new series will be ignored. Limit is currently set at <quota>.
If you exceed the data points per series quota, the following message displays:
Exceeded maximum points per chart, data skipped
If your plotting needs exceed the preceding quotas, create the plots in the Research Environment instead .
Demonstration
For more information about creating custom charts, see Charting .
Adjust Charts
You can manipulate the charts displayed on the backtest results page.
Toggle Charts
To display and hide a chart on the backtest results page, in the Select Chart section, click the name of a chart.
Toggle Chart Series
To display and hide a series on a chart on the backtest results page, click the name of a series at the top of a chart.
Adjust the Display Period
To zoom in and out of a time series chart on the backtest results page, perform either of the following actions:
Click the 1m , 3m , 1y , or All period in the top-right corner of the chart.
Click a point on the chart and drag your mouse horizontally to highlight a specific period of time in the chart.
If you adjust the zoom on a chart, it affects all of the charts.
After you zoom in on a chart, slide the horizontal bar at the bottom of the chart to adjust the time frame that displays.
Resize Charts
To resize a chart on the backtest results page, hover over the bottom-right corner of the chart. When the resize cursor appears,
hold the left mouse button and then drag to the desired size.
Move Charts
To move a chart on the backtest results page, click, hold, and drag the chart title.
Refresh Charts
Refreshing the charts on the backtest results page resets the zoom level on all the charts. If you refresh the charts while your
algorithm is executing, only the data that was seen by the Lean engine after you refreshed the charts is displayed. To refresh
the charts, in the Select Chart section, click the reset icon.
Key Statistics
The backtest results page displays many key statistics to help you analyze the performance of your algorithm.
Overall Statistics
The Overview tab on the backtest results page displays tables for Overall Statistics and Rolling Statistics. The Overall Statistics
table displays the following statistics:
Probabilistic Sharpe Ratio (PSR)
Total Trades
Average Loss
Drawdown
Net Profit
Loss Rate
Profit-Loss Ratio
Beta
Annual Variance
Tracking Error
Total Fees
Lowest Capacity Asset
Sharpe Ratio
Average Win
Compounding Annual Return
Expectancy
Win Rate
Alpha
Annual Standard Deviation
Information Ratio
Treynor Ratio
Estimated Strategy Capacity
Some of the preceding statistics are sampled throughout the backtest to produce a time series of rolling statistics. The time
series are displayed in the Rolling Statistics table.
To download the data from the Overall Statistics and Rolling Statistics tables, see Download Results .
Research Guide
For information about the Research Guide, see Research Guide .
Reports
Backtest reports provide a summary of your algorithm's performance during the backtest period. Follow these steps to generate
one:
1. Open the backtest results page for which you want to generate a report.
2. Click the Report tab.
3. If the project doesn't have a description, enter one and then click Save .
4. Click Download Report .
The report may take a minute to generate.
5. If the IDE says that the report is being generated, repeat step 4.
Orders
The backtest results page displays the orders of your algorithm and you can download them to your local machine.
View in the GUI
To see the orders that your algorithm created, open the backtest results page and then click the Orders tab. If there are more
than 10 orders, use the pagination tools at the bottom of the Orders Summary table to see all of the orders. Click on an individual
order in the Orders Summary table to reveal all of the order events , which include:
Submissions
Fills
Partial fills
Updates
Cancellations
Option contract exercises and expiration
The timestamps in the Order Summary table are based in Eastern Time (ET).
Access the Order Summary CSV
To view the orders data in CSV format, open the backtest results page, click the Orders tab, and then click Download Orders .
The content of the CSV file is the content displayed in the Orders Summary table when the table rows are collapsed. The
timestamps in the CSV file are based in Coordinated Universal Time (UTC).
Access the Orders JSON
To view all of the content in the Orders Summary table, see Download Results .
Access in Jupyter Notebooks
To programmatically analyze orders, call the read_backtest_orders method or the /backtests/orders/read endpoint.
Insights
The backtest results page displays the insights of your algorithm and you can download them to your local machine.
View in the GUI
To see the insights your algorithm emit, open the backtest result page and then click the Insights tab. If there are more than 10
insights, use the pagination tools at the bottom of the Insights Summary table to see all of the insights. The timestamps in the
Insights Summary table are based in Eastern Time (ET).
Download JSON
To view the insights in JSON format, open the backtest result page, click the Insights tab, and then click Download Insights . The
timestamps in the CSV file are based in Coordinated Universal Time (UTC).
Logs
The backtest results page displays the logs of your backtest and you can download them to your local machine. The timestamps
of the statements in the log file are based in your algorithm time zone .
View in the GUI
To see the log file that was created throughout a backtest, open the backtest result page and then click the Logs tab.
To filter the logs, enter a search string in the Filter logs field.
Download Log Files
To download the log file that was created throughout a backtest, follow these steps:
1. Open the backtest result page.
2. Click the Logs tab.
3. Click Download Logs .
Project Files
The backtest results page displays the project files used to run the backtest. To view the files, click the Code tab. By default, the
main.py file displays. To view other files in the project, click the file name and then select a different file from the drop-down
menu.
Share Results
The backtest results page enables you to share your backtest results. You need to make a backtest public in order to share it. To
make a backtest public, in the Share Results section of the backtest results page, click Make Public . Once you make a backtest
public, click Backtest URL to copy a link to the backtest result or click Embed Code to copy an iframe HTML element you can
embed on a website.
The following widget is an example of an embedded backtest result:
CHARTS
CODE
CLONE
STRATEGY REPORT
MEASURED APRICOT CHICKEN
To attach the embedded backtest result to a forum discussion, see Create Discussions or Post Comments .
After you've made your backtest results public, the results are always stored and anyone with the link can access the results. To
make your backtest results private again, click Make Private .
Download Results
You can download the following information from the backtest results page:
Runtime statistics
Charts
The data in the Overview tab
The data in the Orders tab
To download the preceding information, open the backtest results page, click the Overview tab, and then click Download
Results . The timestamps in the file are based in Coordinated Universal Time (UTC).
The information is stored in JSON format. The following code converts the JSON result file into CSV format.
View All Backtests
Follow these steps to view all of the backtests of a project:
1. Open the project that contains the backtests you want to view.
2. In the top-right corner of the IDE, click the Backtest Results icon.
A table containing all of the backtest results for the project is displayed. If there is a play icon to the left of the name, it's a
backtest result . If there is a fast-forward icon next to the name, it's an optimization result .
import json
import pandas as pd
# Load the JSON file.
data = json.loads(open("<YOUR_JSON_FILE>.json", "r", encoding="utf-8").read())
# Recursive function to flatten the JSON and create tuples for multi-index.
def flatten_json(obj, parent_key='', sep='_'):
items = []
for k, v in obj.items():
new_key = f"{parent_key}{sep}{k}" if parent_key else k
if isinstance(v, dict):
items.extend(flatten_json(v, new_key, sep=sep).items())
else:
items.append((new_key, v))
return dict(items)
# Flatten the JSON data.
flat_data = flatten_json(data)
# Split keys into a list for multi-index.
multi_index_tuples = [tuple(k.split('_')) for k in flat_data.keys()]
values = list(flat_data.values())
# Create the multi-index DataFrame.
multi_index = pd.MultiIndex.from_tuples(multi_index_tuples)
df = pd.DataFrame(values, index=multi_index, columns=['Value'])
# Save to CSV.
df.to_csv('results.csv')
PY
3. (Optional) In the top-right corner, select the Show field and then select one of the options from the drop-down menu to
filter the table by backtest or optimization results.
4. (Optional) In the bottom-right corner, click the Hide Error check box to remove backtest and optimization results from the
table that had a runtime error.
5. (Optional) Use the pagination tools at the bottom to change the page.
6. (Optional) Click a column name to sort the table by that column.
7. Click a row in the table to open the results page of that backtest or optimization.
Rename Backtests
We give an arbitrary name (for example, "Smooth Apricot Chicken") to your backtest result files, but you can follow these steps
to rename them:
1. Hover over the backtest you want to rename and then click the pencil icon that appears.
2. Enter the new backtest name and then click OK .
To programmatically set the backtest name, call the set_name method.
For more information, see Set Name and Tags .
Clone Backtests
Hover over the backtest you want to clone, and then click the clone icon that appears to clone the backtest.
A new project is created with the backtest code files.
self.set_name("Backtest Name")
PY
Delete Backtests
Hover over the backtest you want to delete, and then click the trash can icon that appears to delete the backtest.
Out of Sample Period
To reduce the chance of overfitting, organization managers can enforce all backtests must end a certain number of months
before the current date. For example, if you set a one year out-of-sample period, the researchers on your team will not be able
to use the most recent year of data in their backtests. A out-of-sample period is helpful because it leaves you a period to test
your model after your done the development stage. Follow these steps to change the backtest out-of-sample period:
1. Open the organization homepage .
2. Scroll down to the Backtesting Out of Sample Period section.
3. Adjust the out-of-sample period duration or click on "No Holdout Period".
Errors
If a backtest produces more than 700 MB of data, then LEAN can't upload the results and the backtest results page appears
empty.
Backtesting > Results > Portfolio Margin Plots
Results
Portfolio Margin Plots
Introduction
The portfolio margin chart is a stacked area chart where each color represents the proportion of your available margin that a
security consumes over time.
Margin plots can appear counter intuitive and hard to understand. To simplify things, this chart only draws the top 5-10 holdings
at a given point in time. It group the smaller holdings into an area called "Others".
Sample Frequency
In backtests, the chart areas are sampled once a day to produce the plot, so intraday margin usage isn't visible.
Cash vs Margin
When you trade with a margin account, orders consume your cash before borrowing anything on margin. This order of
precedence minimizes the interest charges on your margin loans.
This algorithm produces the following portfolio margin plot. When the initial trades fill, the orders consume available cash before
borrowing anything available from the margin. Consuming available cash first avoids interest charges on the margin-loan and
avoids margin calls during market downturns. This results in a relatively steady line hovering on 50% as there is no free cash to
purchase with, but only the remaining 50% margin.
This algorithm is the same buy-and-hold strategy as the previous algorithm, but it only targets 50% exposure during the initial
trades. The following portfolio margin plot shows the results. This version only consumes 16% of the available margin for each
security, leaving a substantial amount of cash free. As the portfolio value grows, the fraction of the margin that this cash
represents shrinks, showing an upward trending plot as the holdings of AAPL grow in value.
Examples
The following examples provide insight into portfolio margin calculations and show the functionality of the portfolio margin
chart.
Buying Put Options
This algorithm demonstrates that when you buy a put Option contract and you long the underlying asset, it reduces the margin
used by the underlying asset. In the following image, the gray area represents the underlying Equity and the blue area
represents the put Option contract. The algorithm buys SPY on 10/07/2013, buys the put Option contract on 01/02/2014, and
then exercises the contract on 06/21/2014.
Buying Options Spreads
This algorithm demonstrates that when you long and short two Option contracts at the same time, they combine together to
reduce your margin usage. In the following image, the blue area represents a short position in a call Option contract and the
gray area represents a long position in a call Option contract. The algorithm forms a bear call spread one leg at a time, shorting
an in-the-money contract on 02/03/2023 and then longing an out-of-the-money contract on 02/14/2023.
Buying Futures
This algorithm demonstrates that Futures have a fixed, USD-denominated margin requirement. In the following image, the blue
area represents the percentage of the available margin that the Futures contract uses. The algorithm buys the contract on
10/02/2023 and holds it until the end of the backtest. As the total portfolio value decreases during the last two week in the
backtest, the amount of available margin in the portfolio decreases. However, the USD amount of margin used by the Futures
contract is constant, so the proportion of portfolio margin used by the contract increases at the same time.
Buying Forex
This algorithm demonstrates how the 50x leverage available for Forex impacts the portfolio margin chart. In the following image,
the blue area represents the percentage of the available margin that the Forex position uses. The algorithm buys 500%
exposure to the EURUSD pair on 01/02/2018 and holds it until the end of the backtest. The portfolio margin chart shows that only
10% of the available margin is used to enter the trade since 500% / 50 = 10%.
Series Color Changes
The portfolio margin plot updates daily as the backtest runs. However, the color of a specific security in the plot isn't finalized
until the end of the backtest. For instance, the following image shows an example plot during execution:
The following image shows what the plot looks like at the end of the backtest:
Series Names
The series names in the chart are the last known tickers for the respective assets (for example, SPY and XLE). Therefore, if you
trade multiple assets throughout a backtest that share the same last known ticker (for example, FB in 1998 and FB in 2014), the
chart labels them under the same series name.
Backtesting > Debugging
Backtesting
Debugging
Introduction
The debugger is a built-in tool to help you debug coding errors while backtesting. The debugger enables you to slow down the
code execution, step through the program line-by-line, and inspect the variables to understand the internal state of the program.
Breakpoints
Breakpoints are lines in your algorithm where execution pauses. You need at least one breakpoint in your code files to start the
debugger. Open a project to start adjusting its breakpoints.
Add Breakpoints
Click to the left of a line number to add a breakpoint on that line.
Edit Breakpoint Conditions
Follow these steps to customize what happens when a breakpoint is hit:
1. Right-click the breakpoint and then click Edit Breakpoint... .
2. Click one of the options in the following table:
Option Additional Steps Description
Expression
Enter an expression and then press
Enter .
The breakpoint only pauses the
algorithm when the expression is true.
Hit Count Enter an integer and then press Enter .
The breakpoint doesn't pause the
algorithm until its hit the number of
times you specify.
Enable and Disable Breakpoints
To enable a breakpoint, right-click it and then click Enable Breakpoint .
To disable a breakpoint, right-click it and then click Disable Breakpoint .
Follow these steps to enable and disable all breakpoints:
1. In the right navigation menu, click the Run and Debug icon.
2. In the Run and Debug panel, hover over the Breakpoints section and then click the Toggle Active Breakpoints icon.
Remove Breakpoints
To remove a breakpoint, right-click it and then click Remove Breakpoint .
Follow these steps to remove all breakpoints:
1. In the right navigation menu, click the Run and Debug icon.
2. In the Run and Debug panel, hover over the Breakpoints section and then click the Remove All Breakpoints icon.
Launch Debugger
Follow these steps to launch the debugger:
1. Open the project you want to debug.
2. In your project's code files, add at least one breakpoint.
3. Click the Debug icon.
If the Run and Debug panel is not open, it opens when the first breakpoint is hit.
Control Debugger
After you launch the debugger, you can use the following buttons to control it:
Button Name Default Keyboard Shortcut Description
Continue
Continue execution until the
next breakpoint
Step Over Alt+F10
Step to the next line of code
in the current or parent
scope
Step Into Alt+F11
Step into the definition of
the function call on the
current line
Restart Shift+F11 Restart the debugger
Disconnect Shift+F5 Exit the debugger
Inspect Variables
After you launch the debugger, you can inspect the state of your algorithm as it executes each line of code. You can inspect
local variables or custom expressions. The values of variables in your algorithm are formatted in the IDE to improve readability.
For example, if you inspect a variable that references a DataFrame, the debugger represents the variable value as the following:
Local Variables
The Variables section of the Run and Debug panel shows the local variables at the current breakpoint. If a variable in the panel is
an object, click it to see its members. The panel updates as the algorithm runs.
Follow these steps to update the value of a variable:
1. In the Run and Debug panel, right-click a variable and then click Set Value .
2. Enter the new value and then press Enter .
Custom Expressions
The Watch section of the Run and Debug panel shows any custom expressions you add. For example, you can add an
expression to show the current date in the backtest.
Follow these steps to add a custom expression:
1. Hover over the Watch section and then click the plus icon that appears.
2. Enter an expression and then press Enter .
Backtesting > Report
Backtesting
Report
Introduction
Reports provide a summary of your algorithm's performance. They outline key statistics, returns, and performance during
various market crises. You can generate a performance report after your backtest completes and download the report as a PDF.
Key Statistics
The top of the backtest report displays statistics to summarize your algorithm's performance. The following table describes the
key statistics in the report:
Statistic Description
Runtime Days The number of days in the backtest or live trading period.
Turnover
The percentage of the algorithm's portfolio that was
replaced in a given year.
CAGR
The annual percentage return that would be required to
grow a portfolio from its starting value to its ending value.
Markets The asset classes that the algorithm trades.
Trades per day
The total number of trades during the backtest divided by
the number of days in the backtest. Trades per day is an
approximation of the algorithm's trading frequency.
Drawdown
The largest peak to trough decline in an algorithm's equity
curve.
Probabilistic SR
The probability that the estimated Sharpe ratio of an
algorithm is greater than a benchmark (1).
Sharpe Ratio
A measure of the risk-adjusted return, developed by William
Sharpe.
Information Ratio
The amount of excess return from the risk-free rate per unit
of systematic risk.
Strategy Capacity
The maximum amount of money an algorithm can trade
before its performance degrades from market impact.
Returns
The backtest report displays charts to show the algorithm's returns per trade, per day, per month, per year, and the cumulative
returns over the backtest.
Returns per Trade
This chart displays a histogram that shows the distribution of returns per trade over the backtesting period.
Daily Returns
This chart displays the returns of each day. Blue bars represent profitable days and gray bars represent unprofitable days.
Monthly Returns
This chart displays the return of each month. We convert the original equity curve series into a monthly series and calculate the
returns of each month. Green cells represent months with a positive return and red cells represent months with a negative
return. Months that have a greater magnitude of returns are represented with darker cells. Yellow cells represent months with a
relatively small gain or loss. White rectangles represent months that are not included in the backtest period. The values in the
cells are percentages.
Annual Returns
This chart displays the return of each year. We calculate the total return within each year and represent each year with a blue
bar. The red dotted line represents the average of the annual returns.
Cumulative Returns
This chart displays the cumulative returns of your algorithm. The blue line represents your algorithm and the gray line
represents the benchmark.
Asset Allocation
This chart displays a time-weighted average of the absolute holdings value for each asset that entered your portfolio during the
backtest. When an asset has a percentage that is too small to be shown in the pie chart, it is incorporated into an "Others"
category.
Drawdown
This chart displays the peak-to-trough drawdown of your portfolio's equity throughout the backtest period. The drawdown of
each day is defined as the percentage loss since the maximum equity value before the current day. The drawdowns are
calculated based on daily data. The top 5 drawdown periods are marked in the chart with different colors.
Rolling Statistics
The backtest report displays time series for your portfolio's rolling beta and Sharpe ratio .
Rolling Portfolio Beta
This chart displays the rolling portfolio beta over trailing 6 and 12 month periods. The light blue line represents the 6 month
period and the dark blue line represents the 12 month period.
Rolling Sharpe Ratio
This chart displays the rolling portfolio Sharpe ratio over trailing 6 and 12 month periods. The light blue line represents the 6
month period and the dark blue line represents the 12 month period.
Exposure
The backtest report displays time series for your portfolio's overall leverage and your portfolio's long-short exposure by asset
class.
Leverage
This chart displays your algorithm's utilization of leverage over time.
Long-Short Exposure By Asset Class
This chart displays your algorithm's long-short exposure by asset class over time.
Crisis Events
This set of charts displays the cumulative returns of your algorithm and the benchmark during various historical periods. The
blue line represents the cumulative returns of your algorithm and the grey line represents the cumulative return of the
benchmark. The report only contains the crisis event that occurred during your algorithm's backtest period. The following table
shows the crisis events that may be included in your backtest report:
Crisis Name Start Date End Date
DotCom Bubble 2000 2/26/2000 9/10/2000
September 11, 2001 9/5/2001 10/10/2001
U.S. Housing Bubble 2003 1/1/2003 2/20/2003
Global Financial Crisis 2007 10/1/2007 12/1/2011
Flash Crash 2010 5/1/2010 5/22/2010
Fukushima Meltdown 2011 3/1/2011 4/22/2011
U.S. Credit Downgrade 2011 8/5/2011 9/1/2011
ECB IR Event 2012 9/5/2012 10/12/2012
European Debt Crisis 2014 10/1/2014 10/29/2014
Market Sell-Off 2015 8/10/2015 10/10/2015
Recovery 2010-2012 1/1/2010 10/1/2012
New Normal 2014-2019 1/1/2014 1/1/2019
COVID-19 Pandemic 2020 2/10/2020 9/20/2020
Post-COVID Run-up 2020-2021 4/1/2020 1/1/2022
Meme Season 2021 1/1/2021 5/15/2021
Russia Invades Ukraine 2022-2023 2/1/2022 1/1/2024
AI Boom 2022-Present 11/30/2022 Present
Parameters
This section of the report shows the name and value of all the parameters in your project.
Customize Reports
To create custom reports, customize the HTML and CSS.
Customize the Report HTML
The Report / template.html file in the LEAN GitHub repository defines the stucture of the reports you generate. To override the
HTML file, add a report.html file to your project . To include some of the information and charts that are in the default report, use
the report keys in the Report / ReportKey.cs file in the LEAN GitHub repository. For example, to add the Sharpe ratio of your
backtest to the custom HTML file, use {{$KPI-SHARPE}} .
To include the crisis event plots in your report, add the {{$HTML-CRISIS-PLOTS}} key and then define the structure of the
individual plots inside of <!--crisis and crisis--> . Inside of this comment, you can utilize the {{$TEXT-CRISIS-TITLE}} and
{{$PLOT-CRISIS-CONTENT}} keys. For example, the following HTML is the default format for each crisis plot:
<!--crisis
<div class="col-xs-4">
<table class="crisis-chart table compact">
<thead>
<tr>
<th style="display: block; height: 75px;">{{$TEXT-CRISIS-TITLE}}</th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding:0;">
<img src="{{$PLOT-CRISIS-CONTENT}}">
</td>
</tr>
</tbody>
</table>
</div>
crisis--&gt
To include the algorithm parameters in your report, add the {{$PARAMETERS}} key and then define the HTML element inside of
<!--parameters and parameters--> . Inside of this comment, you can use special keys {{$KEY<parameterIndex>}} and
{{$VALUE<parameterIndex>}} , which represent the key and value of a single parameter. For example, the following HTML is
the default format for the parameters element:
<!--parameters
<tr>
<td class = "title"> {{$KEY0}} </td><td> {{$VALUE0}} </td>
<td class = "title"> {{$KEY1}} </td><td> {{$VALUE1}} </td>
</tr>
parameters--&gt
In the preceding example, {{$KEY0}} is the name of the first parameter in the algorithm and {{$VALUE0}} is its value.
Customize the Report CSS
The Report / css / report.css file in the LEAN GitHub repository defines the style of the reports you generate. To override the
stylesheet, add a report.css file to your project .
Backtesting > Engine Performance
Backtesting
Engine Performance
Introduction
A set of benchmark algorithms are periodically run to test the status and speed of the Lean master branch. View the Lean
Performance Benchmark page to see the results. The chart at the top of the page shows the data points per second for each of
the benchmark algorithms. The table at the bottom of the page shows the benchmark algorithms that are run to produce the
results.
Datasets
Datasets
Datasets are a stream of data points you use in your algorithms to make trading decisions and fill orders. Our Dataset Market is a
portal where we aggregate datasets for you to use in your algorithms. Our Dataset Market includes price, fundamental, and
alternative datasets. Consider using fundamental and alternative datasets to incorporate more information in your trading
decisions. Fundamental and alternative datasets contain information that is not present in the price. Price data is commonly
researched for trading ideas, so you may find it easier to discover alpha in other types of datasets.
The Dataset Market enables you to easily load datasets into your trading algorithms for use in the cloud or locally. The datasets
come configured ready to integrate into your research and backtesting without any need for cleaning. The datasets in our
market are vetted by the QuantConnect team to be high-quality, contain actionable information, and be free of survivorshipbias. Our Dataset Market is growing quickly. New datasets are added frequently.
Navigating the Market
The lay of the land
Categories
Diverse types of datasets available
Data Issues
Data isn't always perfect
Misconceptions
It might not be a data issue
Licensing
Ways to access the data
Vendors
People who provide datasets
QuantConnect
Data Provider
Alpaca
Brokerage Data Provider
Charles Schwab
Brokerage Data Provider
Interactive Brokers
Brokerage Data Provider
Polygon
Data Provider
Samco
Brokerage Data Provider
TradeStation
Brokerage Data Provider
Tradier
Brokerage Data Provider
Zerodha
Brokerage Data Provider
See Also
Dataset Market
Purchasing Datasets
Contributing Datasets
Datasets > Navigating the Market
Datasets
Navigating the Market
Introduction
Datasets are a stream of data points you use in your algorithms to make trading decisions and fill orders. Our Dataset Market is a
portal where we curate datasets available for use in your algorithms. It includes price, fundamental, and alternative datasets.
Consider using fundamental and alternative datasets to incorporate more information in your trading decisions. Fundamental
and alternative datasets contain information that is not present in the price. Price data is commonly researched for trading ideas,
so you may find it easier to discover alpha in other types of datasets.
The Dataset Market enables you to easily load datasets into your trading algorithms for use in the cloud or locally. The datasets
come configured ready to integrate into your research and backtesting without any need for cleaning. The datasets in our
market are vetted by the QuantConnect team to be high-quality, contain actionable information, and be free of survivorshipbias. Our Dataset Market is growing quickly, as new datasets are added frequently.
The Dataset Market is a place where you can view, subscribe to, and download datasets. We provide an example algorithm for
each dataset that you can clone to easily get started using the new dataset. We also provide an example research notebook for
most datasets to demonstrate how to use the dataset in the Research Environment. You can always view the dataset reviews to
learn about the experience other members have had using the dataset. This page explains the layout of the Dataset Market to
help you navigate the marketplace.
View All Listings
The Dataset Market displays all our supported datasets. To view the page, in the top navigation bar, click Data.
Each dataset displays the name, description, coverage, start date, and price of the dataset. Coverage is the number of assets,
securities, contracts, currency pairs, or articles that are included in the dataset. To view the listing page of a dataset, click the
dataset.
You can search the market by applying filters or searching for keywords.
Filter Listings by Category
Click the Category: All field and then select a category from the drop-down menu to only display datasets in that category.
Filter Listings by Delivery Options
Click the Delivery Options: All field and then select an option from the drop-down menu to only display datasets with that
delivery option.
Search for Keywords
Enter keywords in the search bar to only display datasets that contain those keywords.
Homepage
The homepage of a dataset listing displays everything that you need to get started using the dataset. The following table
describes the tabs on the homepage:
Tab Description
About High-level overview of the dataset and the data provider
Documentation Instructions on using the dataset in backtests and the Research Environment
Research A demonstration research notebook of analyzing the dataset
Examples Full example algorithms that use the dataset
Licenses A list of licenses available for the dataset
CLI Command generator to download the dataset with the CLI
Pricing The price to access the dataset in the cloud or on your local machine
Data Explorer A table to inspect the dataset files and report data issues
The following table describes the sections displayed under the About tab for most datasets:
Tab Description
Introduction High-level overview of what the dataset includes, who it's created by, and how it's created.
About the Provider Description of the data provider and a link to their website.
Getting Started The line(s) of code that you need to use the dataset in algorithms.
Data Summary A table that displays the dataset's start date, coverage, resolution, density, and timezone.
Example Applications A list of ideas on using the dataset in your algorithm.
Data Point Attributes
A set of widgets that display the factors in the dataset, the class members of objects that
you use when accessing the dataset, and enumeration values that you can use to customize
the dataset.
Pricing The price to access the dataset in the cloud or on your local machine.
Reviews Reviews from members who have purchased the dataset.
Sidebar
The sidebar of the dataset listing provides a brief summary of the dataset. The following table describes the summary content:
Tab Description
Pricing The number of licensing options available
Start Date The date of the first data point.
Coverage
The number of assets, securities, contracts, currency pairs, or articles that the dataset
includes.
Delivery Methods The various delivery methods the dataset supports.
About the Provider A link to the data provider's website.
Documentation
The Documentation tab on a dataset listing demonstrates how to use the dataset. The documentation covers requesting the
data, accessing the data in your algorithm, and performing history requests. We provide documentation in C# and Python so
you can easily integrate the dataset into your algorithms, regardless of the programming language you use.
The Documentation tab also has a Data Point Attributes section to show the dataset's attributes. If an attribute has a custom data
type, you can click the attribute to view the attributes of the custom data type.
Factor Research
Some dataset listings have a Research tab that displays an analysis of the data point attributes in the dataset. Follow these steps
to clone the example research notebook of a dataset:
1. Log in to the Algorithm Lab.
2. Open a dataset listing page .
3. Click the Research tab, if available.
4. Click Clone This Notebook .
5. Click Clone Algorithm .
Examples
The Examples tab on a dataset listing shows how to use the dataset in a trading algorithm. We provide examples in C# and
Python for both of the classic and framework algorithm designs. Copy-paste these example algorithms to jumpstart your
strategy research. Consider adjusting the strategy to make it your own or using the parameter optimization feature to improve
the performance of the algorithms.
Licenses
The Licensing tab shows the available licenses for the dataset. Each dataset comes with its own licensing requirements,
depending on the data vendor. For more information about licensing types, see Licensing .
Reviews
The bottom of the dataset listing page shows reviews published by QuantConnect members. You can sort, filter, and write
dataset reviews.
Sort Reviews
Open a dataset listing page , click the Most Recent field, and then select a metric from the drop-down menu to sort the reviews
by that metric.
Filter Reviews
Open a dataset listing page , click the All field, and then select a number of stars from the drop-down menu to only display the
reviews with that rating.
Write Reviews
Follow these steps to write a review:
1. Log in to the Algorithm Lab.
2. Open a dataset listing page .
3. At the bottom of the page, select a number of stars to give your review.
4. Write your review.
5. Click Submit Review .
Datasets > Categories
Datasets
Categories
Introduction
Dataset categories are a way to identify different types of datasets in our Dataset Market. We provide many price, fundamental,
and alternative datasets for you to use in your research and trading. Datasets that include factors outside of the security price
are less researched, so they may have more alpha to discover. Incorporate alternative datasets into your algorithms so that you
can make more informed trading decisions.
Geospatial
Geospatial data is data related to objects that have a position in the world.
Commerce
Commerce data is data on customer and business behavior.
Financial Market
Financial market data is data on the trading activity on exchanges.
Consumer
Consumer data is data on all aspects of consumers, including online shopping behaviors, consumer demographics, and
consumer attitudes.
B2B
Business-to-business (B2B) data is data on businesses that sell goods and services to other businesses.
Transport and Logistics
Transport and logistics data is data on the transportation of goods and the logistics of the transportation.
Environmental
Environmental data is data on the state of the environment, including meteorological data, biodiversity data, and pollution data.
Credit Rating
Credit rating data is data on the financial position of individuals and businesses.
Real Estate
Real estate data is data on residential and commercial real estate, including ownership data, real estate listing data, and real
estate demographic data.
Web
Web data is data on the behavior of internet users.
Legal
Legal data is data on the law, including new regulations, government website data, and litigation history.
Health Care
Healthcare data is data on patient-doctor visits, including claims data, fitness wearables data, and health record data.
Entertainment
Entertainment data is data on the media consumption preferences and behaviors of consumers.
Energy
Energy data is data on energy production, distribution, and consumption.
Industry
Industry data is data on various groupings in the economy.
Political
Political data is data that's collected on political activity, including election votes and political party policies.
News and Events
News and events data is data that's collected from news providers regarding current events.
Datasets > Data Issues
Datasets
Data Issues
Introduction
Data issues are incorrect or missing values in a dataset. These issues are generally a result of human error or from mistakes in
the data collection process. Data issues can be reported by any QuantConnect member. When data issues are reported and
verified, our Data Team works to quickly resolve them. Thanks to the communal efforts of the QuantConnect members, the
QuantConnect data is reviewed and fixed by over 250,000 people, giving you a very high-quality source of data.
Common Issues
Data issues can occur in both historical and live data providers. Some common examples of data issues include the following:
Missing or incorrect values
Splits and dividends
Listings and delistings
Ticker changes
View Current Issues
To view the list of current data issues, log in to the Algorithm Lab and then, in the left navigation bar, click Datasets > Data
Issues . Before you report a new issue, review the list of current issues to ensure that the issue is not already reported. The
number of open data issues can sometimes be large, but our Data Team works on resolving them as quickly as possible while
prioritizing the most important ones.
Report New Issues
When all of the QuantConnect members report the data issues that they find, we can ensure the datasets are high quality for
everyone. The easier it is for our Data Team to detect and reproduce the issues you report, the faster we can resolve them. If
you encounter an issue with live data, email us a description of the issue. If you find an issue with the historical data of a
dataset, follow these steps to report it:
1. Log in to the Algorithm Lab.
2. Open the Data Explorer Issues page.
3. On the Data Explorer page, fill out the form.
4. Follow these steps to attach a backtest or notebook that demonstrates the issue:
1. Click Attach Backtest .
2. Click the Project field and then select the project from the drop-down menu.
3. Click the Backtest field and then select the backtest from the drop-down menu.
5. Click Publish .
Datasets > Misconceptions
Datasets
Misconceptions
Introduction
Some data issues are reported that aren't actually data issues. Instead, they are from a misunderstanding of how the data is
collected, timestamped, formatted, and normalized. These misunderstandings are caused by assumptions that the data should
be the same across different platforms, should have the same timezones, should be timestamped a certain way, and should be
normalized the same as other data sources.
Cross-Platform Discrepancies
You may find our data can sometimes be slightly different from the data that's displayed on other platforms. Most of the
differences occur because our data is institutional quality while a lot of the other platforms use a cheaper alternative. We use
the Consolidated Tape Association (CTA) and Unlisted Trading Privileges (UTP) tick feeds, which cover the entire US tick feed.
In contrast, most charting websites use the Better Alternative Trading System (BATS), which has very permissive display
policies but only covers about 6-7% of the total market volume . Our tick feed doesn't include over-the-counter (OTC) trades,
but the data on other platforms like Yahoo Finance include OTC trades.
Timezone Differences
Datasets all have different timezones. Most price datasets are timestamped in Eastern Time (ET). However, Future markets have
more exotic timezones, depending on where the Future contract is trading. QuantConnect allows the raw data to be in different
timezones. For US Equities, the timezone is ET. For Forex prices, the timezone is Coordinated Universal Time (UTC). In contrast,
other charting platforms may display data with ET timestamps. Forex uses UTC, but CFD uses timezones relative to each of the
CFD products that lists. QuantConnect accurately reflects all of these timezones from the relative markets that they're trading.
Misaligned Timestamps
Every piece of data has a period. Some data is near-instantaneous, like tick data. Other data has a longer period, like second,
minute, hour, and daily bars. QuantConnect delivers this data to your algorithms at the end of the period to ensure that
lookahead bias doesnʼt occur. When you look at the time property of your algorithm, the period has already ended, so it looks as
if the data is offset by one period. To compare the timestamps of our data to other data, use the time property of the current
bar. The time property of the bar is the start of the bar and the end_time property is the end of the bar. If you use Python and
request historical data, the time index in the DataFrame that's returned maps to the end_time of the respective bar. For more
information about timestamps, see Time Modeling .
Data Normalization
The data normalization mode defines how historical data is adjusted to accommodate for splits, dividends, and continuous
Future contract roll overs. When you compare the data in the Dataset Market to data that's hosted on other platforms, the data
may have different values because a different data normalization mode is being used to adjust the data. Ensure datasets are
using the same normalization mode before reporting data issues. The most common way to recognize this bug is by comparing
the two price series and seeing them significantly deviate in the past. The following data normalization modes are available:
Adjusted Prices
By default, LEAN adjusts US Equity data for splits and dividends to produce a smooth price curve. We use the entire split and
dividend history to adjust historical prices. This process ensures you get the same adjusted prices, regardless of the backtest
end date.
Backtest differences occur when you run backtests before a split or dividend occurs in live trading and then run the same
backtest after it occurs. The second time you run the backtest, the adjusted prices will be different so it can cause different
backtest results. The difference can be significant in large universes because of multiple corporate actions and the cummulative
effect of orders with a small difference.
Opening and Closing Auctions
The opening and closing price of the day is set by very specific opening and closing auction ticks. When a stock like Apple is
listed, itʼs listed on Nasdaq. The open auction tick on Nasdaq is the price thatʼs used as the official open of the day. NYSE,
BATS, and other exchanges also have opening auctions, but the only official opening price for Apple is the opening auction on
the exchange where it was listed.
We set the opening and closing prices of the first and last bars of the day to the official auction prices. This process is used for
second, minute, hour, and daily bars for the 9:30 AM and 4:00 PM Eastern Time (ET) prices. In contrast, other platforms might
not be using the correct opening and closing prices.
The official auction prices are usually emitted 2-30 seconds after the market open and close. We do our best to use the official
opening and closing prices in the bars we build, but the delay can be so large that there isn't enough time to update the opening
and closing price of the bar before it's injected into your algorithms. For example, if you subscribe to second resolution data, we
wait until the end of the second for the opening price but most second resolution data wonʼt get the official opening price. If you
subscribe to minute resolution data, we wait until the end of the minute for the opening auction price. Most of the time, youʼll get
the actual opening auction price with minute resolution data, but there are always exceptions. Nasdaq and NYSE can have
delays in publishing the opening auction price, but we donʼt have control over those issues and we have to emit the data on time
so that you get the bar you are expecting.
Live and Backtesting Differences
In live trading, bars are built using the exchange timestamps with microsecond accuracy. This microsecond-by-microsecond
processing of the ticks can mean that the individual bars between live trading and backtesting can have slightly different ticks.
As a result, it's possible for a tick to be counted in different bars between backtesting and live trading, which can lead to bars
having slightly different open, high, low, close, and volume values.
There is a delay in when new live data is available for backtesting. It's normally available after 24-48 hours. If you need to
closely monitor new data, use live paper trading .
Datasets > Licensing
Datasets
Licensing
Introduction
You can license datasets in the Dataset Market to use in the cloud for live trading and research or to download locally. We have
contracts with the data providers in the Dataset Market that define the costs of each license. All of the datasets can be used in
QuantConnect Cloud. There are some free licenses, but we can't freely redistribute most of the datasets.
Free
We strive to make as many datasets available for free to use in the cloud and to download locally as possible. We also list
proprietary datasets that are available for license using our cloud or download paid licensing. Most price data is free for use in
the cloud.
Cloud
If you have a Cloud license for a dataset, you can access the dataset for research, backtests, and live trading in the Algorithm
Lab. The cost of the license is added to your monthly bill, which you can pay with your organization's credit card or
QuantConnect Credit balance. With one Cloud license for a dataset, all of the members in your organization can access the
dataset in the cloud.
Add Cloud Access
You need an organization above the Free tier to purchase cloud access to datasets.
Follow these steps to add cloud access to datasets:
1. Log in to the Algorithm Lab.
2. Open the listing page of a dataset for which you want to gain cloud access.
3. On the dataset listing page, click the Pricing tab.
4. Under the Cloud Access section, click SUBSCRIBE .
5. On the Pricing page, click Proceed to Checkout .
Remove Cloud Access
Follow these steps to remove cloud access to datasets:
1. Open the organization homepage .
2. Click Edit Plan .
3. On the pricing page, click the Customize Plan > Build Your Own Pack > Data tab.
4. In the Datasets Subscriptions section, next to the name of the dataset you want to remove, click Added .
5. Click Proceed to Checkout .
Download
If you have a Download license, you can store datasets on your local machine. This download is for the licensed organization's
internal LEAN use only and cannot be redistributed or converted in any format. If you study the data and produce some charts,
you may share images of the charts online if the original data can't be to reconstructed from the image. The cost of the license
depends on the dataset and it's calculated on a per-file or per-day basis. For more information about downloading datasets, see
Downloading Data . If you bulk download datasets, you can download historical data packages or daily updates. In most cases,
you need both.
Datasets > Vendors
Datasets
Vendors
Introduction
We welcome submissions of new datasets by data companies. Review our submission process to learn how to get your dataset
listed on QuantConnect.
Submission Criteria
Datasets must meet the following criteria to be considered for the Datasets Market:
A well-defined dataset with a clear and static vision for the data to minimize churn or changes.
Robust ticker and security links to ensure the tickers are tracked well through time. ISIN, FIGI, or point-in-time tickers are
supported.
Sufficient organizational funding to ensure at least 1 year of operation.
Reliable API with no dead links or 502 errors.
Consistent delivery schedule with data delivered on time and in time for trading.
Consistent data format with notifications and lead time on data format updates.
At least 1 year of historical data.
Free of survivorship bias.
Good documentation of the dataset.
If the dataset is alternative data, in addition to the criteria above, the dataset must practice the Alternative Investment Standards
defined by the non-profit Investment Data Standards Organization (IDSO). The Alternative Investment Standards outline the
rules and best practices for collecting and distributing alternative datasets. For example, the AWS Web Crawling Best Practices
publication states that “when a web crawler encounters a robots.txt file on a website, it parses the instructions and adjusts its
crawling behavior accordinglyˮ.
Review Process
The dataset review process is in place to ensure that your dataset matches the submission criteria. The review process can take
several weeks. If your dataset is accepted, we'll begin integrating it into the Datasets Market. If your dataset is rejected, we'll
provide feedback to help you get the dataset accepted.
Contributing Datasets
We encourage you to integrate your own datasets. To integrate your dataset, see the Contributing Datasets tutorial. The
integration process only takes about 1 day of engineering.
Give Free Trials
Follow these steps to give a free trial of your dataset to a QuantConnect organization:
1. Log in to the Algorithm Lab.
2. Open the Dataset Market .
3. On the Datasets page, click the dataset that you want to give as a trial.
4. In the right sidebar of the dataset listing, click Dashboard .
5. On the dataset dashboard page, click give trial .
6. In the Give Trial to Organization window, enter the expriation date of the trial and then click OK .
7. In the Organization Owners Email window, enter the email address of the member who owns the organization that you
want to grant the trial.
8. If the email address you entered owns mulitple organizations, in the Select target organization window, select an
organization from the drop-down menu and then click OK .
9. Click OK .
Contacting Our Team
If you want to discuss integrating your dataset into the Datasets Market, contact us . We look forward to working with you so
that we can provide QuantConnect members with access to more high-quality datasets.
Datasets > QuantConnect
Datasets
QuantConnect
The QuantConnect data provider provides a stream of trades and quotes to your trading algorithm during live execution. Live
data enables you to make real-time trades and update the value of the securities in your portfolio.
When you deploy a live algorithm, you can use the QuantConnect data provider, a third-party data provider, or both. If you use
multiple data providers, the order you select them in defines their order of precedence in Lean. For example, if you set
QuantConnect as the first provider and Polygon as the second provider, Lean only uses the Polygon data provider for securities
that aren't available from the QuantConnect data provider. This configuration makes it possible to use our data provider for
Equity universe selection and then place Options trades on the securities in the universe.
US Equities
Crypto
Crypto Futures
CFD
Forex
Futures
Alternative Data
See Also
Dataset Market
Live Trading
Datasets > QuantConnect > US Equities
QuantConnect
US Equities
Introduction
The QuantConnect data provider provides a stream of US Equity trades and quotes to your trading algorithm during live
execution. Live data enables you to make real-time trades and update the value of the securities in your portfolio.
The QuantConnect data provider also provides a live stream of corporate actions (US Equity Security Master), daily updates on
company fundamentals (US Fundamentals), and the number of shares that are available for short sellers to borrow (US Equities
Short Availability).
Sourcing
The QuantConnect data provider consolidates US Equity market data across all of the exchanges. Over-the-Counter (OTC)
trades are excluded. The data is powered by the Securities Information Processor (SIP), so it has 100% market coverage. In
contrast, free platforms that display data feeds like the Better Alternative Trading System (BATS) only have about 6-7% market
coverage .
We provide live splits, dividends, and corporate actions for US companies. We deliver them to your algorithm before the trading
day starts.
Universe Selection
The QuantConnect data provider enables you to create a dynamic universe of US Equities.
Fundamental Universe
The live data for fundamental universe selection arrives at 6/7 AM Eastern Time (ET), so fundamental universe selection runs for
live algorithms between 7 and 8 AM ET. This timing allows you to place trades before the market opens. Don't schedule
anything for midnight because the universe selection data isn't ready yet.
ETF Constituent Universe
The QuantConnect data provider enables you to create a universe of securities to match the constituents of an ETF. For more
information about ETF universes, see ETF Constituents Selection .
Bar Building
We aggregate ticks to build bars.
# Run universe selection asynchronously to speed up your algorithm.
self.universe_settings.asynchronous = True
self.add_universe(self._select_fundamental)
# Request an async universe of stocks that match the ETF constituents of SPY
self.universe_settings.asynchronous = True
spy = self.add_equity("SPY").symbol
self.add_universe(self.universe.etf(spy, self.universe_settings, self._etf_constituents_filter))
PY
PY
Discrepancies
In live trading, bars are built using the exchange timestamps with microsecond accuracy. This microsecond-by-microsecond
processing of the ticks can mean that the individual bars between live trading and backtesting can have slightly different ticks.
As a result, it's possible for a tick to be counted in different bars between backtesting and live trading, which can lead to bars
having slightly different open, high, low, close, and volume values.
Opening and Closing Auctions
The opening and closing price of the day is set by very specific opening and closing auction ticks. When a stock like Apple is
listed, itʼs listed on Nasdaq. The open auction tick on Nasdaq is the price thatʼs used as the official open of the day. NYSE,
BATS, and other exchanges also have opening auctions, but the only official opening price for Apple is the opening auction on
the exchange where it was listed.
We set the opening and closing prices of the first and last bars of the day to the official auction prices. This process is used for
second, minute, hour, and daily bars for the 9:30 AM and 4:00 PM Eastern Time (ET) prices. In contrast, other platforms might
not be using the correct opening and closing prices.
The official auction prices are usually emitted 2-30 seconds after the market open and close. We do our best to use the official
opening and closing prices in the bars we build, but the delay can be so large that there isn't enough time to update the opening
and closing price of the bar before it's injected into your algorithms. For example, if you subscribe to second resolution data, we
wait until the end of the second for the opening price but most second resolution data wonʼt get the official opening price. If you
subscribe to minute resolution data, we wait until the end of the minute for the opening auction price. Most of the time, youʼll get
the actual opening auction price with minute resolution data, but there are always exceptions. Nasdaq and NYSE can have
delays in publishing the opening auction price, but we donʼt have control over those issues and we have to emit the data on time
so that you get the bar you are expecting.
Excluded Ticks
The bar-building process can exclude ticks. If a tick is excluded, its volume is aggregated in the bar but its price is not
aggregated in the bar. Ticks are excluded if any of the following statements are true:
The tick is suspicious.
The tick is from the FINRA exchange and meets our price and volume thresholds.
The trade has none of the following included TradeConditionFlags and at least one of the following excluded
TradeConditionFlags :
TradeConditionFlags Status Description
REGULAR Included
A trade made without stated conditions is
deemed the regular way for settlement on
the third business day following the
transaction date.
FORM_T Included
Trading in extended hours enables
investors to react quickly to events that
typically occur outside regular market
hours, such as earnings reports.
However, liquidity may be constrained
during such Form T trading, resulting in
wide bid-ask spreads.
CASH Included
A transaction that requires delivery of
securities and payment on the same day
the trade takes place.
EXTENDED_HOURS Included
Identifies a trade that was executed
outside of regular primary market hours
and is reported as an extended hours
trade.
NEXT_DAY Included
A transaction that requires the delivery of
securities on the first business day
following the trade date.
OFFICIAL_CLOSE Included
Indicates the "official" closing value
determined by a Market Center. This
transaction report will contain the market
center generated closing price.
OFFICIAL_OPEN Included
Indicates the 'Official' open value as
determined by a Market Center. This
transaction report will contain the market
center generated opening price.
CLOSING_PRINTS Included
The transaction that constituted the
trade-through was a single priced closing
transaction by the Market Center.
OPENING_PRINTS Included
The trade that constituted the tradethrough was a single priced opening
transaction by the Market Center.
INTERMARKET_SWEEP Excluded
The transaction that constituted the
trade-through was the execution of an
order identified as an Intermarket Sweep
Order.
TRADE_THROUGH_EXEMPT Excluded Denotes whether or not a trade is exempt
(Rule 611).
ODD_LOT Excluded Denotes the trade is an odd lot less than a
100 shares.
The quote has a size of less than 100 shares.
The quote has none of the following included QuoteConditionFlags and at least one of the following excluded
QuoteConditionFlags :
The quote has one of the following QuoteConditionFlags :
QuoteConditionFlags Status Description
CLOSING Included
Indicates that this quote was the last
quote for a security for that Participant.
NEWS_DISSEMINATION Included
Denotes a regulatory trading halt when
relevant news influencing the security is
being disseminated. Trading is
suspended until the primary market
determines that an adequate publication
or disclosure of information has
occurred.
NEWS_PENDING Included
Denotes a regulatory Trading Halt due to
an expected news announcement, which
may influence the security. An Opening
Delay or Trading Halt may be continued
once the news has been disseminated.
TRADING_RANGE_INDICATI
ON
Included
Denotes the probable trading range (Bid
and Offer prices, no sizes) of a security
that is not Opening Delayed or Trading
Halted. The Trading Range Indication is
used prior to or after the opening of a
security.
ORDER_IMBALANCE Included
Denotes a non-regulatory halt condition
where there is a significant imbalance of
buy or sell orders.
RESUME Included
Indicates that trading for a Participant is
no longer suspended in a security that
had been Opening Delayed or Trading
Halted.
REGULAR Excluded
This condition is used for the majority of
quotes to indicate a normal trading
environment.
SLOW Excluded
This condition is used to indicate that the
quote is a Slow Quote on both the bid and
offer sides due to a Set Slow List that
includes high price securities.
GAP Excluded
While in this mode, auto-execution is not
eligible, the quote is then considered
manual and non-firm in the bid and offer,
and either or both sides can be traded
through as per Regulation NMS.
OPENING_QUOTE Excluded
This condition can be disseminated to
indicate that this quote was the opening
quote for a security for that Participant.
FAST_TRADING Excluded
For extremely active periods of short
duration. While in this mode, the UTP
Participant will enter quotations on a best
efforts basis.
RESUME Excluded
Indicate that trading for a Participant is no
longer suspended in a security which had
been Opening Delayed or Trading Halted.
In the preceding tables, Participant refers to the entities on page 19 of the Consolidated Tape System Multicast Output Binary
Specification .
Suspicious Ticks
Tick price data is raw and unfiltered, so it can contain a lot of noise. If a tick is not tradable, we flag it as suspicious. This
process makes the bars a more realistic representation of what you could execute in live trading. If you use tick data, avoid
using suspicious ticks in your algorithms as informative data points. We recommend only using tick data if you understand the
risks and are able to perform your own tick filtering. Ticks are flagged as suspicious in the following situations:
The tick occurs below the best bid or above the best ask
This image shows a tick that occurred above the best ask price of a security. The green line represents the best ask of the
security, the blue line represents the best bid of the security, and the red dots represent trade ticks. The ticks between the
best bid and ask occur from filling hidden orders. The tick that occurred above the best ask price is flagged as suspicious.
The tick occurs far from the current market price
This image shows a tick that occurred far from the price of the security. The red dots represent trade ticks. The tick that
occurred far from the market price is flagged as suspicious.
The tick occurs on a dark pool
The tick is rolled back
The tick is reported late
Delivery
Most live trading algorithms run on co-located servers racked in Equinix. Co-location reduces several factors that can interfere
with your algorithm, including downtime from internet outages, equipment repairs, and natural disasters.
Live data takes time to travel from the source to your algorithm. The QuantConnect latencies vary depending on the data
provider, but for US Equities, we have a latency of 5-40 milliseconds. A much more significant source of latency is the round trip
order times from brokers, which can vary from 100ms to 5 seconds. QuantConnect is not intended for high-frequency trading,
but we have integrations to high-speed brokers if you need.
Historical Data
When you request historical data or run warm-up , the amount of historical data you can access depends on the resolution of
your data subscriptions. The following table shows the amount of trailing historical data you can access for each data
resolution:
Resolution Available History
Daily All historical data
Hour All historical data
Minute 1 year
Second 2 months
Tick 1 month
Pricing
The QuantConnect data provider serves US Equities data for free.
Datasets > QuantConnect > Crypto
QuantConnect
Crypto
Introduction
The QuantConnect data provider provides a stream of Crypto trades and quotes to your trading algorithm during live execution.
Live data enables you to make real-time trades and update the value of the securities in your portfolio.
Sourcing
The QuantConnect data provider uses WebSockets to gather Crypto market data from the following sources:
Binance & Binance US
Bitfinex
Bybit
Coinbase
Kraken
Universe Selection
The QuantConnect data provider enables you to create a dynamic universe of Crypto securities. The live data for Crypto
universe selection arrives at 4 PM Coordinated Universal Time (UTC), so universe selection runs for live algorithms between 4
PM and 4:30 PM. Don't schedule anything for midnight because the universe selection data isn't ready yet.
To view an example for each Crypto market, see the Universe Selection section of the Crypto market in the Dataset Market .
Bar Building
We aggregate ticks to build bars.
In live trading, bars are built using the exchange timestamps with microsecond accuracy. This microsecond-by-microsecond
processing of the ticks can mean that the individual bars between live trading and backtesting can have slightly different ticks.
As a result, it's possible for a tick to be counted in different bars between backtesting and live trading, which can lead to bars
having slightly different open, high, low, close, and volume values.
Delivery
Most live trading algorithms run on co-located servers racked in Equinix. Co-location reduces several factors that can interfere
with your algorithm, including downtime from internet outages, equipment repairs, and natural disasters.
Live data takes time to travel from the source to your algorithm. The QuantConnect latencies vary depending on the data
provider, but for US Equities, we have a latency of 5-40 milliseconds. A much more significant source of latency is the round trip
order times from brokers, which can vary from 100ms to 5 seconds. QuantConnect is not intended for high-frequency trading,
but we have integrations to high-speed brokers if you need.
# Request an async Bitfinex crypto universe universe with a selection filter.
self.universe_settings.asynchronous = True
self.add_universe(CryptoUniverse.bitfinex(self._universe_selection_filter))
PY
Historical Data
When you request historical data or run warm-up , the amount of historical data you can access depends on the resolution of
your data subscriptions. The following table shows the amount of trailing historical data you can access for each data
resolution:
Resolution Available History
Daily All historical data
Hour All historical data
Minute 1 year
Second 2 months
Tick 1 month
Pricing
The QuantConnect data provider serves Crypto data for free.
Datasets > QuantConnect > Crypto Futures
QuantConnect
Crypto Futures
Introduction
The QuantConnect data provider provides a stream of Crypto Futures trades and quotes to your trading algorithm during live
execution. Live data enables you to make real-time trades and update the value of the securities in your portfolio.
Sourcing
The QuantConnect data provider uses WebSockets to gather Crypto Futures market data from Binance.
Bar Building
We aggregate ticks to build bars.
In live trading, bars are built using the exchange timestamps with microsecond accuracy. This microsecond-by-microsecond
processing of the ticks can mean that the individual bars between live trading and backtesting can have slightly different ticks.
As a result, it's possible for a tick to be counted in different bars between backtesting and live trading, which can lead to bars
having slightly different open, high, low, close, and volume values.
Delivery
Most live trading algorithms run on co-located servers racked in Equinix. Co-location reduces several factors that can interfere
with your algorithm, including downtime from internet outages, equipment repairs, and natural disasters.
Live data takes time to travel from the source to your algorithm. The QuantConnect latencies vary depending on the data
provider, but for US Equities, we have a latency of 5-40 milliseconds. A much more significant source of latency is the round trip
order times from brokers, which can vary from 100ms to 5 seconds. QuantConnect is not intended for high-frequency trading,
but we have integrations to high-speed brokers if you need.
Historical Data
When you request historical data or run warm-up , the amount of historical data you can access depends on the resolution of
your data subscriptions. The following table shows the amount of trailing historical data you can access for each data
resolution:
Resolution Available History
Daily All historical data
Hour All historical data
Minute 1 year
Second 2 months
Tick 1 month
Pricing
The QuantConnect data provider serves Crypto Futures data for free.
Datasets > QuantConnect > CFD
QuantConnect
CFD
Introduction
The QuantConnect data provider provides a stream of CFD trades and quotes to your trading algorithm during live execution.
Live data enables you to make real-time trades and update the value of the securities in your portfolio.
Sourcing
The QuantConnect data provider consolidates CFD market data from OANDA.
Bar Building
We aggregate ticks to build bars.
In live trading, bars are built using the exchange timestamps with microsecond accuracy. This microsecond-by-microsecond
processing of the ticks can mean that the individual bars between live trading and backtesting can have slightly different ticks.
As a result, it's possible for a tick to be counted in different bars between backtesting and live trading, which can lead to bars
having slightly different open, high, low, close, and volume values.
Delivery
Most live trading algorithms run on co-located servers racked in Equinix. Co-location reduces several factors that can interfere
with your algorithm, including downtime from internet outages, equipment repairs, and natural disasters.
Live data takes time to travel from the source to your algorithm. The QuantConnect latencies vary depending on the data
provider, but for US Equities, we have a latency of 5-40 milliseconds. A much more significant source of latency is the round trip
order times from brokers, which can vary from 100ms to 5 seconds. QuantConnect is not intended for high-frequency trading,
but we have integrations to high-speed brokers if you need.
Historical Data
When you request historical data or run warm-up , the amount of historical data you can access depends on the resolution of
your data subscriptions. The following table shows the amount of trailing historical data you can access for each data
resolution:
Resolution Available History
Daily All historical data
Hour All historical data
Minute 1 year
Second 2 months
Tick 1 month
Pricing
The QuantConnect data provider serves CFD data for free.
Datasets > QuantConnect > Forex
QuantConnect
Forex
Introduction
The QuantConnect data provider provides a stream of Forex pair trades and quotes to your trading algorithm during live
execution. Live data enables you to make real-time trades and update the value of the securities in your portfolio.
Sourcing
The QuantConnect data provider consolidates Forex market data from OANDA.
Bar Building
We aggregate ticks to build bars.
In live trading, bars are built using the exchange timestamps with microsecond accuracy. This microsecond-by-microsecond
processing of the ticks can mean that the individual bars between live trading and backtesting can have slightly different ticks.
As a result, it's possible for a tick to be counted in different bars between backtesting and live trading, which can lead to bars
having slightly different open, high, low, close, and volume values.
Delivery
Most live trading algorithms run on co-located servers racked in Equinix. Co-location reduces several factors that can interfere
with your algorithm, including downtime from internet outages, equipment repairs, and natural disasters.
Live data takes time to travel from the source to your algorithm. The QuantConnect latencies vary depending on the data
provider, but for US Equities, we have a latency of 5-40 milliseconds. A much more significant source of latency is the round trip
order times from brokers, which can vary from 100ms to 5 seconds. QuantConnect is not intended for high-frequency trading,
but we have integrations to high-speed brokers if you need.
Historical Data
When you request historical data or run warm-up , the amount of historical data you can access depends on the resolution of
your data subscriptions. The following table shows the amount of trailing historical data you can access for each data
resolution:
Resolution Available History
Daily All historical data
Hour All historical data
Minute 1 year
Second 2 months
Tick 1 month
Pricing
The QuantConnect data provider serves Forex data for free.
Datasets > QuantConnect > Futures
QuantConnect
Futures
Introduction
The QuantConnect data provider provides a stream of Futures trades, quotes, and open interest to your trading algorithm during
live execution. Live data enables you to make real-time trades and update the value of the securities in your portfolio.
Sourcing
The QuantConnect data provider consolidates Futures market data across the following markets:
CBOT
CME
COMEX
ICE
NYMEX
The data is powered by the Chicago Mercantile Exchange (CME).
The data provider doesn't include the CFE market. For Futures.Indices.VIX , use a combination of the QuantConnect and IB
data provider. For more details about this option, see Hybrid Data Provider .
Bar Building
We aggregate ticks to build bars.
In live trading, bars are built using the exchange timestamps with microsecond accuracy. This microsecond-by-microsecond
processing of the ticks can mean that the individual bars between live trading and backtesting can have slightly different ticks.
As a result, it's possible for a tick to be counted in different bars between backtesting and live trading, which can lead to bars
having slightly different open, high, low, close, and volume values.
Delivery
Most live trading algorithms run on co-located servers racked in Equinix. Co-location reduces several factors that can interfere
with your algorithm, including downtime from internet outages, equipment repairs, and natural disasters.
Live data takes time to travel from the source to your algorithm. The QuantConnect latencies vary depending on the data
provider, but for US Equities, we have a latency of 5-40 milliseconds. A much more significant source of latency is the round trip
order times from brokers, which can vary from 100ms to 5 seconds. QuantConnect is not intended for high-frequency trading,
but we have integrations to high-speed brokers if you need.
Historical Data
When you request historical data or run warm-up , the amount of historical data you can access depends on the resolution of
your data subscriptions. The following table shows the amount of trailing historical data you can access for each data
resolution:
Resolution Available History
Daily All historical data
Hour All historical data
Minute 1 year
Second 2 months
Tick 1 month
Pricing
The QuantConnect data provider serves Futures data for free.
Datasets > QuantConnect > Alternative Data
QuantConnect
Alternative Data
Introduction
The QuantConnect data provider can stream live alternative data into your algorithms to help you make informed trading
decisions.
Sourcing
The QuantConnect data provider sources alternative data directly from data vendors . To view all of the integrated data vendors,
see the Dataset Market .
Delivery
The delivery schedule of alternative data depends on the specific dataset you're using. We inject the data into your algorithms
when the vendor provides the data. For most alternative datasets, the data is updated on a daily or hourly basis. Some datasets,
like the Tiingo News Feed or Benzinga News Feed , include a live stream. In these cases, we deliver the data as a live stream to
your algorithm.
Most live trading algorithms run on co-located servers racked in Equinix. Co-location reduces several factors that can interfere
with your algorithm, including downtime from internet outages, equipment repairs, and natural disasters.
Live data takes time to travel from the source to your algorithm. The latency of the alternative data depends on the specific
dataset you're using.
Historical Data
When you request historical data or run warm-up , the amount of historical data you can access depends on the resolution of
your data subscriptions. The following table shows the amount of trailing historical data you can access for each data
resolution:
Resolution Available History
Daily All historical data
Hour All historical data
Minute 1 year
Second 2 months
Tick 1 month
Pricing
Refer to the Dataset Market listings.
Datasets > Alpaca
Datasets
Alpaca
Introduction
Alpaca was founded by Yoshi Yokokawa and Hitoshi Harada in 2015 as a database and machine learning company. In 2018,
Alpaca Securities LLC (Alpaca Securities) became a registered US broker-dealer with the Financial Industry Regulatory Authority
( FINRA ) with the mission to "open financial services to everyone on the planet". In 2022, Alpaca Crypto LLC (Alpaca Crypto)
became a registered Money Services Business (MSB) with the Financial Crimes Enforcement Network ( FinCEN ). Alpaca
provides access to trading Equities, Options, and Crypto for clients in over 30 countries. Alpaca also delivers custody, clearing,
execution, and billing on behalf of registered advisors.
The Alpaca data provider serves Equity, Equity Option, and Crypto prices directly from Alpaca's Market Data API . This page
explains our integration with their API and its functionality.
Sourcing
The Alpaca data provider sources data directly from Alpaca's Market Data API . If you use this data provider, Alpaca only
provides the security price data. QuantConnect Cloud provides the following auxiliary datasets:
US Equity Security Master
US Equity Option Universe
Universe selection datasets
Non-streaming alternative datasets
Universe Selection
Universe selection is available with the Alpaca data provider.
The Alpaca data provider can stream data for up to 30 assets on the Basic plan. If your algorithm adds more than the quota,
LEAN logs an error message from Alpaca. To increase the quota, purchase the Algo Trader Plus plan from Alpaca.
Bar Building
The data feed is a stream of asset prices collected by WebSockets and distributed to algorithms on the platform.
Alternative Data
Third-party data providers support most alternative datasets, except data that streams real-time intraday data. Streaming
datasets, like the Tiingo News Feed and Benzinga News Feed , require the QuantConnect data provider.
Hybrid Data Provider
When you deploy a live algorithm with the Alpaca brokerage , you can use a third-party data provider, the Alpaca data provider,
or both. If you use multiple data providers, the order you select them in defines their order of precedence in Lean. For example,
if you set QC as the first provider and Alpaca as the second provider, Lean only uses the Alpaca data provider for securities that
self.universe_settings.asynchronous = True
self.add_universe(self.fundamental_universe_selection)
PY
aren't available from the QC data provider. This configuration makes it possible to use our data provider for Equity universe
selection and then place Options trades on the securities in the universe.
Historical Data
The historical data that's available from the Alpaca data provider for history requests and warm-up periods depends on your
Alpaca data plan. For more information about each plan, see the Data page on the Alpaca website.
Pricing
The Alpaca data feed is free for Alpaca subscription accounts. The Basic plan has a quota of 30 assets and 200 API calls per
minute. The Algo Trader Plus plan has higher quotas. To view the latest prices of the plans, see the Data page on the Alpaca
website.
Datasets > Charles Schwab
Datasets
Charles Schwab
Introduction
The Charles Schwab Corporation was founded by Charles R. Schwab in 1971. Charles Schwab provides access to trading
Equities, Options, Index Options, and other assets for clients with no account or trade minimums, or hidden fees .
The Charles Schwab data provider serves Equity, Equity Option, Index, and Index Option prices directly from Charles Schwab's
Trader API . This page explains our integration with their API and its functionality.
Sourcing
The Charles Schwab data provider sources data directly from Charles Schwab's Trader API . If you use this data provider,
Charles Schwab only provides the security price data. QuantConnect Cloud provides the following auxiliary datasets:
US Equity Security Master
US Equity Option Universe
US Index Option Universe
Universe selection datasets
Non-streaming alternative datasets
Universe Selection
Universe selection is available with the Charles Schwab data provider.
Bar Building
The data feed is a stream of asset prices collected by WebSockets and distributed to algorithms on the platform.
Alternative Data
Third-party data providers support most alternative datasets, except data that streams real-time intraday data. Streaming
datasets, like the Tiingo News Feed and Benzinga News Feed , require the QuantConnect data provider.
Hybrid Data Provider
When you deploy a live algorithm with the Charles Schwab brokerage , you can use a third-party data provider, the Charles
Schwab data provider, or both. If you use multiple data providers, the order you select them in defines their order of precedence
in Lean. For example, if you set QC as the first provider and Charles Schwab as the second provider, Lean only uses the Charles
Schwab data provider for securities that aren't available from the QC data provider. This configuration makes it possible to use
our data provider for Equity universe selection and then place Options trades on the securities in the universe.
Pricing
The Charles Schwab data feed is free for Charles Schwab subscription accounts.
self.universe_settings.asynchronous = True
self.add_universe(self.fundamental_universe_selection)
PY
Datasets > Interactive Brokers
Datasets
Interactive Brokers
Introduction
Interactive Brokers (IB) was founded by Thomas Peterffy in 1993 with the goal to "create technology to provide liquidity on
better terms. Compete on price, speed, size, diversity of global products and advanced trading tools". IB provides access to
trading Equities, ETFs, Options, Futures, Future Options, Forex, CFDs, Gold, Warrants, Bonds, and Mutual Funds for clients in
over 200 countries and territories with no minimum deposit. IB also provides paper trading, a trading platform, and educational
services.
The Interactive Brokers (IB) data provider serves asset prices from IB's Trader Workstation API . This page explains our
integration with their data API and its functionality.
Sourcing
The IB data provider sources data directly from IB's Trader Workstation API . If you use this data provider, IB only provides the
security price data. QuantConnect Cloud provides the following auxiliary datasets:
US Equity Security Master
US Futures Security Master
US Equity Option Universe
US Index Option Universe
Universe selection datasets
Non-streaming alternative datasets
Universe Selection
Universe selection is available with the IB data provider.
The universe selection data comes from our Dataset Market, not the TWS market scanners . Universe selection with the IB data
provider occurs around 6-7 AM Eastern Time (ET) on Tuesday to Friday and at 2 AM ET on Sunday. Universe selection data isn't
available when the IB servers are closed. To check the IB server status, see the Current System Status page on the IB website.
The IB data provider can stream data for up to 100 assets by default, but IB may let you stream more than 100 assets based on
your commissions and equity value. For more information about quotas from IB, see the Market Data Pricing Overview page on
the IB website. If your algorithm adds more than the your quota, LEAN logs an error message from IB. To increase the quota,
purchase a Quote Booster from IB.
Bar Building
The data is a summarized snapshot of the trades and quotes at roughly 300 milliseconds per snapshot.
Alternative Data
self.universe_settings.asynchronous = True
self.add_universe(self.fundamental_universe_selection)
PY
Third-party data providers support most alternative datasets, except data that streams real-time intraday data. Streaming
datasets, like the Tiingo News Feed and Benzinga News Feed , require the QuantConnect data provider. The hybrid
QuantConnect- IB data provider supports streaming datasets.
Hybrid Data Provider
When you deploy a live algorithm with the IB brokerage , you can use a third-party data provider, the IB data provider, or both. If
you use multiple data providers, the order you select them in defines their order of precedence in Lean. For example, if you set
QC as the first provider and IB as the second provider, Lean only uses the IB data provider for securities that aren't available
from the QC data provider. This configuration makes it possible to use our data provider for Equity universe selection and then
place Options trades on the securities in the universe. If you use a third-party data provider, the assets that you subscribe to
don't contribute to the IB data limit .
Historical Data
If you get historical data from IB through a history request or a warm-up period , the historical data has the following
characteristics:
Second resolution data is limited to six months of history.
The historical data excludes delisted Equities and expired Options.
The historical data excludes expired Futures after two years.
The following quotas are in place for tick and second resolution historical data:
You can have up to 50 simultaneous requests.
You can make up to 60 requests within any 10-minute period.
In the preceding quotas, TradeBar and QuoteBar data count as separate requests. For example, if you request TradeBar and
QuoteBar data for SPY, it counts as two requests.
For more information about historical data from IB, see Historical Data Limitations in the IB documentation.
Pricing
To use IB data in your algorithms, subscribe to IB market data . We support all of the IB data subscriptions that are related to the
securities and markets we support . Members usually subscribe to the following IB market data:
US Securities Snapshot and Futures Value Bundle
US Equity and Options Add-On Streaming Bundle
CFE Enhanced Top of Book (L1 for VIX Futures)
CME S&P Indexes (L1 for SPX and NDX)
CBOE Streaming Market Indexes (L1 for VIX Index)
To see the latest prices, check the Market Data Pricing Overview page on the IB website. IB can take up to 24 hours to process
subscription requests. So after you subscribe to data, you need to wait 24 hours before you can use it in your algorithms. When
you subscribe to data, IB only assigns your data subscription to one of your accounts. If you want to assign the subscription to a
different account, for example, a paper trading account instead of a live trading account, then contact IB.
Datasets > Polygon
Datasets
Polygon
Introduction
Polygon was founded by Quinton Pike in 2017 with the goal to "break down the barriers that have traditionally limited access to
high-quality financial data for all". Polygon provides institutional-grade Equity, Option, Index, Forex, and Crypto data for
business and educational purposes.
The Polygon data provider serves asset prices from Polygon. Instead of using the data from QuantConnect or your brokerage,
you can use data from Polygon if you're deploying a live algorithm and have an API key. To get an API key, see the API Keys
page on the Polygon website. This page explains our integration with Polygon and its functionality.
To view the implementation of the Polygon integration, see the Lean.DataSource.Polygon repository .
QuantConnect Cloud currently only supported streaming Polygon data during live trading. To download Polygon for backtesting,
research, and optimizations, use the CLI .
Supported Datasets
The Polygon data provider sources asset price data directly from Polygon. Our integration supports securities from the following
asset classes:
US Equity
US Equity Options
US Indices
US Index Options
To supplement the asset price data from Polygon, QuantConnect Cloud provides the following auxiliary datasets from the
Dataset Market:
US Equity Security Master
US Equity Option Universe
US Index Option Universe
Universe selection datasets
Non-streaming alternative datasets
For more information about the Polygon data source, see the Polygon API documentation .
Universe Selection
When you trade live on QuantConnect Cloud with the Polygon data provider, QuantConnect provides the universe selection
data.
Alternative Data
The alternative datasets in QuantConnect Cloud works seamlessly with the data from Polygon.
Research
The Polygon data provider is not currently supported for research in QuantConnect Cloud.
Backtesting
The Polygon data provider is not currently supported for backtesting in QuantConnect Cloud.
Optimization
The Polygon data provider is not currently supported for optimizations in QuantConnect Cloud.
Live Trading
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live trading algorithm that uses the Polygon data provider:
1. Open the project that you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click your brokerage from the drop-down menu.
4. Enter the required brokerage authentication information.
For more information about the required information for each brokerage, see the Deploy Live Algorithms section of your
brokerage documentation .
5. In the Data Provider section of the deployment wizard, click Show .
6. Click the Data Provider 1 field and then click Polygon from the drop-down menu.
7. Enter your Polygon API Key.
8. Click Save .
9. (Optional) If your brokerage supports exisiting cash and position holdings , add them.
10. (Optional) Set up notifications .
11. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
12. Click Deploy .
Mutiple Data Providers
When you deploy a live algorithm , you can add multiple data providers. If you use multiple data providers, the order you select
them in defines their order of precedence in Lean. For example, if you set QuantConnect as the first provider and Polygon as the
second provider, Lean only uses the Polygon data provider for securities that aren't available from the QuantConnect data
provider. This configuration makes it possible to use QuantConnect data provider for Equity universe selection and use Polygon
for Options on the securities in the universe.
Historical Data
The historical data that's available from the Polygon data provider for history requests and warm-up periods depends on your
Polygon API package. For more information about each package, see the Simple Pricing page on the Polygon website.
Pricing
To view the prices of the Polygon API packages, see the Simple Pricing page on the Polygon website.
Datasets > Samco
Datasets
Samco
Introduction
Samco was founded by Jimeet Modi in 2015 with a mission of providing retail investors access to sophisticated financial
technology that can assist retail investors in creating wealth at a low cost. Samco provides access to India Equities for clients in
India with no minimum balance. Samco also provides stock ratings, mutual funds, and a mini-portfolio investment platform.
The Samco data provider serves live asset prices from Samco's Trade API . This page explains our integration with their API and
its functionality.
Sourcing
The Samco data provider sources data directly from Samco's Trade API . If you use this data provider, Samco only provides the
security price data. QuantConnect Cloud provides the following auxiliary datasets:
India Equity Security Master
Non-streaming alternative datasets
Universe Selection
Universe selection isn't available with the Samco data provider.
Bar Building
The Samco data provider consolidates prices and quotes across all of the Indian exchanges. For a complete list of exchange
and securities, see the ScripMaster file from the StockNote API documentation.
Alternative Data
Third-party data providers support most alternative datasets, except data that streams real-time intraday data. Streaming
datasets, like the Tiingo News Feed and Benzinga News Feed , require the QuantConnect data provider.
Historical Data
For more information about this historical data that's available from the Samco data provider for history requests and warm-up
periods , see the Samco Trade API documentation .
Pricing
The Samco data provider is free. To access it, you just need an active Samco account.
Datasets > TradeStation
Datasets
TradeStation
Introduction
TradeStation was founded by brothers William (Bill) and Rafael (Ralph) Cruz in 1982 as Omega Research, Inc. In 2001, the
company converted itself from a trading software company to an online securities brokerage and renamed itself "TradeStation"
with the mission to "create the ultimate trading experience". TradeStation provides access to trading Equities, Equity Options,
and Futures for clients in over 150 markets, 34 countries, and 27 currencies. TradeStation also delivers custody, clearing,
execution, and billing on behalf of registered advisors.
The TradeStation data provider serves Equity, Equity Option, and Futures prices directly from TradeStation's MarketData API .
This page explains our integration with their API and its functionality.
Sourcing
The TradeStation data provider sources data directly from TradeStation's MarketData API . If you use this data provider,
TradeStation only provides the security price data. QuantConnect Cloud provides the following auxiliary datasets:
US Equity Security Master
US Futures Security Master
US Equity Option Universe
Universe selection datasets
Non-streaming alternative datasets
Universe Selection
Universe selection is available with the TradeStation data provider.
Bar Building
The data feed is a stream of asset prices collected by WebSockets and distributed to algorithms on the platform.
Alternative Data
Third-party data providers support most alternative datasets, except data that streams real-time intraday data. Streaming
datasets, like the Tiingo News Feed and Benzinga News Feed , require the QuantConnect data provider.
Hybrid Data Provider
When you deploy a live algorithm with the TradeStation brokerage, you can use a third-party data provider, the TradeStation
data provider, or both. If you use multiple data providers, the order you select them in defines their order of precedence in Lean.
For example, if you set QC as the first provider and TradeStation as the second provider, Lean only uses the TradeStation data
provider for securities that aren't available from the QC data provider. This configuration makes it possible to use our data
provider for Equity universe selection and then place Options trades on the securities in the universe.
self.universe_settings.asynchronous = True
self.add_universe(self.fundamental_universe_selection)
PY
Historical Data
If you get historical data from TradeStation through a history request or a warm-up period , you can't request tick or second
resolution data.
Pricing
To view the latest prices, see the Data page on the TradeStation website.
Datasets > Tradier
Datasets
Tradier
Introduction
Tradier was founded by Dan Raju, Peter Laptewicz, Jason Barry, Jeyashree Chidambaram, and Steve Agalloco in 2012 with the
goal to "deliver a choice of low-cost, high-value brokerage services to traders". Tradier provides access to trading Equities and
Options for clients in over 250 countries and territories with no minimum deposit for cash accounts . Tradier also delivers
custody, clearing, execution, and billing on behalf of registered advisors.
The Tradier data provider serves Equity and Option prices directly from Tradier's Market Data API . This page explains our
integration with their API and its functionality. If you deploy to the demo environment, Tradier doesn't offer streaming market
data due to exchange restrictions related to delayed data, so use the QuantConnect data provider .
Sourcing
The Tradier data provider sources data directly from Tradier's Market Data API . If you use this data provider, Tradier only
provides the security price data. QuantConnect Cloud provides the following auxiliary datasets:
US Equity Security Master
US Equity Option Universe
Universe selection datasets
Non-streaming alternative datasets
For more information about the data source, see the Tradier API documentation .
Universe Selection
Universe selection is available with the Tradier data provider.
Bar Building
The data feed is a stream of asset prices collected by WebSockets and distributed to algorithms on the platform.
Alternative Data
Third-party data providers support most alternative datasets, except data that streams real-time intraday data. Streaming
datasets, like the Tiingo News Feed and Benzinga News Feed , require the QuantConnect data provider.
Hybrid Data Provider
When you deploy a live algorithm with the Tradier brokerage , you can use a third-party data provider, the Tradier data provider,
or both. If you use multiple data providers, the order you select them in defines their order of precedence in Lean. For example,
if you set QC as the first provider and Tradier as the second provider, Lean only uses the Tradier data provider for securities
that aren't available from the QC data provider. This configuration makes it possible to use our data provider for Equity universe
selection and then place Options trades on the securities in the universe.
self.universe_settings.asynchronous = True
self.add_universe(self.fundamental_universe_selection)
PY
Historical Data
Historical data isn't available for expired Options from the Tradier data provider. For more information about the data that's
available, see Get Historical Pricing in the Tradier documentation.
Pricing
The Tradier data feed is free for Tradier subscription accounts. If you have a free Tradier account, you may have to pay
inactivity and maintenance fees. If you have less than 2, 000intotalaccountvalueandlessthan2executedtradesin1year, theinactivityfeeis
50. If you have less than 2 executed trades per month, the international account monthly maintenance fee is $20. To view the
latest prices, see the Pricing page on the Tradier website.
Datasets > Zerodha
Datasets
Zerodha
Introduction
Zerodha was founded by Nithin Kamath in 2010 with the goal to break all barriers that traders and investors face in India in terms
of cost, support, and technology. Zerodha provides access to India Equities for clients in India with no minimum balance
required. Zerodha also provides a mutual fund investment platform and an interactive portfolio dashboard.
The Zerodha data provider serves asset prices from Zerodha's Kite Connect API . This page explains our integration with their
API and its functionality.
Sourcing
The Zerodha data provider sources data directly from Zerodha's Kite Connect API . If you use this data provider, Zerodha only
provides the security price data. QuantConnect Cloud provides the following auxiliary datasets:
India Equity Security Master
Non-streaming alternative datasets
For more information about the data source, see the Kite Connect API documentation .
Universe Selection
Universe selection isn't available with the Zerodha data provider.
Bar Building
The data provider consolidates prices and quotes across all of the Indian exchanges.
Alternative Data
Third-party data providers support most alternative datasets, except data that streams real-time intraday data. Streaming
datasets, like the Tiingo News Feed and Benzinga News Feed , require the QuantConnect data provider.
Historical Data
The historical data that's available from the Zerodha data provider for history requests and warm-up periods is in a minute,
hourly, or daily resolution. For more information about the data that's available, see Historical candle data in the Kite Connect 3
API documentation.
Pricing
The Zerodha data feed costs ₹2000/month for retail users. To view the latest prices, see the What are the charges for KITE
APIs? page on the Zerodha website.
Live Trading
Live Trading
A live algorithm is an algorithm that trades in real-time with real market data. QuantConnect enables you to run your algorithms
in live mode with real-time market data. Deploy your algorithms using QuantConnect because our infrastructure is battle-tested.
The algorithms that our members create are run on co-located servers and the trading infrastructure is maintained at all times
by our team of engineers. It's common for members to achieve 6-months of uptime with no interruptions.
Getting Started
Learn the basics
Brokerages
All the supported brokerages
Deployment
Run on co-located servers
Notifications
Stay informed on algorithm decisions
Results
Intervene in your live algorithms
Algorithm Control
Differences between backtesting and live trading
Reconciliation
What could happen
Risks
See Also
Adding Notifications
Set Up Paper Trading
Live Trading > Getting Started
Live Trading
Getting Started
Introduction
A live algorithm is an algorithm that trades in real-time with real market data. QuantConnect enables you to run your algorithms
in live mode with real-time market data. Deploy your algorithms using QuantConnect because our infrastructure is battle-tested.
The algorithms that our members create are run on co-located servers and the trading infrastructure is maintained at all times
by our team of engineers. It's common for members to achieve 6-months of uptime with no interruptions.
Deploy Live Algorithms
The following video demonstrates how to deploy live paper trading algorithms:
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live paper trading algorithm:
1. Open the project that you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Paper Trading from the drop-down menu.
4. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
5. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
6. (Optional) Follow these steps to start the algorithm with existing cash holdings ( see video ):
1. In the Algorithm Cash State section, click Show .
2. Click Add Currency .
3. Enter the currency ticker (for example, USD or BTC) and a quantity.
7. (Optional) Follow these steps to start the algorithm with existing position holdings ( see video ):
1. In the Algorithm Holdings State section, click Show .
2. Click Add Holding .
3. Enter the symbol ID, symbol, quantity, and average price.
8. (Optional) Set up notifications .
9. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
10. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
To deploy a live algorithm with a different brokerage, see the Deploy Live Algorithms section of the brokerage integration
documentation .
Stop Live Algorithms
The live trading results page has a Stop button to immediately stop your algorithm from executing. When you stop a live
algorithm, your portfolio holdings are retained. Stop your algorithm if you want to perform any of the following actions:
Update your project's code files
Upgrade the live trading node
Update the settings you entered into the deployment wizard
Place manual orders through your brokerage account instead of the web IDE
Furthermore, if you receive new securities in your portfolio because of a reverse merger, you also need to stop and redeploy the
algorithm.
LEAN actively terminates live algorithms when it detects interference outside of the algorithm's control to avoid conflicting race
conditions between the owner of the account and the algorithm, so avoid manipulating your brokerage account and placing
manual orders on your brokerage account while your algorithm is running. If you need to adjust your brokerage account
holdings, stop the algorithm, manually place your trades, and then redeploy the algorithm.
Follow these steps to stop your algorithm:
1. Open your algorithm's live results page .
2. Click Stop .
3. Click Stop again.
Liquidate Live Algorithms
The live results page has a Liquidate button that acts as a "kill switch" to sell all of your portfolio holdings. If your algorithm has a
bug in it that caused it to purchase a lot of securities that you didn't want, this button let's you easily liquidate your portfolio
instead of placing many manual trades. When you click the Liquidate button, if the market is open for an asset you hold, the
algorithm liquidates it with market orders. If the market is not open, the algorithm places market on open orders. After the
algorithm submits the liquidation orders, it stops executing.
Follow these steps to liquidate your positions:
1. Open your algorithm's live results page .
2. Click Liquidate .
3. Click Liquidate again.
Update Live Algorithms
If you need to adjust your algorithm's project files or parameter values , stop your algorithm, make your changes, and then
redeploy your algorithm. You can't adjust your algorithm's code or parameter values while your algorithm executes.
When you stop and redeploy a live algorithm, your project's live results is retained between the deployments. To clear the live
results history, clone the project and then redeploy the cloned version of the project.
To update parameters in live mode, add a Schedule Event that downloads a remote file and uses its contents to update the
parameter values.
Clear Live Algorithms History
When you stop and redeploy a live algorithm, your project's live results is retained between the deployments. To clear the live
results history, clone the project and then redeploy the cloned version of the project.
On-Premise Live Algorithms
For information about on-premise live trading with Local Platform , see Getting Started .
Get Deployment Id
To get the live deployment Id, open the log file and enter "Launching analysis for" into the search bar. The log file shows all of
the live deployment Ids for the project. An example deployment Id is L-6bf91128391608d0728ff90b81bfca41. If you have
deployed the project multiple times, use the most recent deployment Id in the log file.
Automate Deployments
If you have multiple deployments, use a notebook in the Research Enviroment to programmatically deploy, stop or liquidate
algorithms.
def initialize(self):
self.parameters = { }
if self.live_mode:
def download_parameters():
content = self.download(url_to_remote_file)
# Convert content to self.parameters
self.schedule.on(self.date_rules.every_day(), self.time_rules.every(timedelta(minutes=1)),
download_parameters)
PY
Live Trading > Brokerages
Live Trading
Brokerages
Brokerages supply a connection to the exchanges so that you can automate orders using LEAN.
QuantConnect Paper Trading
US Equities, FOREX, CFD, Crypto, Futures, & Future Options
Interactive Brokers
US Equities, Equity Options, FOREX, Futures, Future Options, Index, Index Options, & CFD
TradeStation
US Equities, Equity Options, Futures, Index, & Index Options
Alpaca
US Equities, Equity Options, & Crypto
Charles Schwab
US Equities, Equity Options, Index, & Index Options
Binance
Crypto & Crypto Futures
ByBit
Crypto & Crypto Futures
Tradier
US Equities & Equity Options
Kraken
Crypto
Coinbase
Crypto
Bitfinex
Crypto
Bloomberg EMSX
US Equities, Equity Options, Futures, & Index Options
Trading Technologies
Futures
Wolverine
US Equities
FIX Connections
Financial Information eXchange
CFD and FOREX Brokerages
CFD & FOREX
Unsupported Brokerages
Request New Additions
See Also
Adding Notifications
Set Up Paper Trading
Live Trading > Brokerages > QuantConnect Paper Trading
Brokerages
QuantConnect Paper Trading
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
QuantConnect Paper Trading lets you run live, real-time data into your algorithm but execute trades using fictional capital.
Instead of your orders being routed to an exchange when you're paper trading, your order fills are simulated. Use paper trading
to test your algorithm without risking real money and to ensure your backtest wasn't overfit before deploying with real money.
You can use the paper trading brokerage without needing to sign up for a real brokerage account. If you don't set a brokerage
model in your algorithm with the set_brokerage_model method, the paper trading brokerage uses the DefaultBrokerageModel
to simulate trades.
To view the implementation of the QuantConnect Paper Trading brokerage, see PaperBrokerage.cs in the LEAN GitHub
repository. To view the implementation of the backtesting brokerage, see BacktestingBrokerage.cs in the LEAN GitHub
repository.
Account Types
The QuantConnect Paper Trading brokerage supports cash and margin accounts. To set the account type in an algorithm, see
the paper trading brokerage model documentation .
If you pass a different BrokerageName to the set_brokerage_model method, the new brokerage model defines the account types
that are available.
Asset Classes
The QuantConnect Paper Trading brokerage supports the following asset classes:
US Equities
Crypto
Forex
CFD
Futures
Future Options
If you set the brokerage model to a model other than the DefaultBrokerageModel , the new brokerage model defines the asset
classes you can trade.
Data Feeds
We can only provide paper trading on the assets for which we have a live data provider .
Orders
The following sections describe how the DefaultBrokerageModel handles orders. If you set the brokerage model to a different
model, the new brokerage model defines how orders are handled.
Order Types
The following table describes the available order types for each asset class that the DefaultBrokerageModel supports:
Order Type US Equity Crypto
Crypto
Futures
Forex CFD Futures
Futures
Options
Market
Limit
Limit if
touched
Stop
market
Stop limit
Market on
open
Market on
close
Combo
market
Combo limit
Combo leg
limit
Exercise
Option
In live trading, Option orders require a third-party data provider that supports Options. To view the data providers that support
Options, see Datasets .
Time In Force
The DefaultBrokerageModel supports the following TimeInForce instructions:
DAY
good_til_canceled
good_til_date
Updates
The DefaultBrokerageModel supports order updates .
Handling Splits
If you're using raw data normalization and you have active orders with a limit, stop, or trigger price in the market for a US Equity
when a stock split occurs, the following properties of your orders automatically adjust to reflect the stock split:
Quantity
Limit price
Stop price
Trigger price
Fees
The following table shows the fees that the DefaultBrokerageModel charges for each of the supported asset classes:
Asset Class Fee
Equities 0.005/sharewitha1 minimum fee
Crypto $0
Forex $0
CFDs $0
Futures $1.85/contract
Future Options $1.85/contract
There is no fee to exercise Option contracts.
If you set the brokerage model to a model other than the DefaultBrokerageModel , the new brokerage model defines the order
fees.
To see the fee models that the DefaultBrokerageModel uses, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you set the
brokerage model to a different model, the new brokerage model defines how margin is modeled. If you have more than $25,000
in your brokerage account, you can use the PatternDayTradingMarginModel to make use of the 4x intraday leverage and 2x
overnight leverage available on most brokerages from the PDT rule .
Slippage
Orders through the DefaultBrokerageModel do not experience slippage in backtests or paper trading. For more information
about the slippage model the DefaultBrokerageModel uses, see Slippage .
Fills
The DefaultBrokerageModel fills market orders immediately and completely. When available, bid and ask spread will be used
for the fill prices.
To view how we model realistic order fills, see Fills .
Settlements
If you trade with a margin account, trades settle immediately
To view how we model settlement for paper trades, see Settlement .
Brokerage Models
The QuantConnect Paper Trading brokerage uses the DefaultBrokerageModel by default, but you can use any of the brokerage
models .
Deposits and Withdrawals
The QuantConnect Paper Trading brokerage supports deposits and withdrawals.
Demo Algorithm
The following algorithm demonstrates the functionality of the DefaultBrokerageModel :
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live paper trading algorithm:
1. Open the project that you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Paper Trading from the drop-down menu.
4. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
5. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
6. (Optional) Follow these steps to start the algorithm with existing cash holdings ( see video ):
1. In the Algorithm Cash State section, click Show .
2. Click Add Currency .
3. Enter the currency ticker (for example, USD or BTC) and a quantity.
7. (Optional) Follow these steps to start the algorithm with existing position holdings ( see video ):
1. In the Algorithm Holdings State section, click Show .
2. Click Add Holding .
3. Enter the symbol ID, symbol, quantity, and average price.
# Deposit 100 units of account currency and withdraw 1 ETH.
self.porfolio.cash_book.add(self.account_currency, 100)
self.porfolio.cash_book.add("ETH", -1)
 Charts  Statistics  Code Clone Algorithm
PY
8. (Optional) Set up notifications .
9. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
10. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Interactive Brokers
Brokerages
Interactive Brokers
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Interactive Brokers (IB) was founded by Thomas Peterffy in 1993 with the goal to "create technology to provide liquidity on
better terms. Compete on price, speed, size, diversity of global products and advanced trading tools". IB provides access to
trading Equities, ETFs, Options, Futures, Future Options, Forex, CFDs, Gold, Warrants, Bonds, and Mutual Funds for clients in
over 200 countries and territories with no minimum deposit. IB also provides paper trading, a trading platform, and educational
services.
To view the implementation of the IB brokerage integration, see the Lean.Brokerages.InteractiveBrokers repository .
Account Types
The IB API does not support the IBKR LITE plan. You need an IBKR PRO plan. Individual and Financial Advisor (FA) accounts are
available.
Individual Accounts
IB supports cash and margin accounts. To set the account type in an algorithm, see the IB brokerage model documentation .
FA Accounts
IB supports FA accounts for Trading Firm and Institution organizations. FA accounts enable certified professionals to use a
single trading algorithm to manage several client accounts. If your account code starts with F, FA, or I, then you have an FA
account. For more information about FA accounts, see Financial Advisors .
Create an Account
You need to open an IBKR Pro account to deploy algorithms with IB. The IB API does not support IBKR Lite accounts. To create
an IB account, see the Open an Account page on the IB website.
You need to activate IBKR Mobile Authentication (IB Key) to deploy live algorithms with your brokerage account. After you open
your account, follow the installation and activation instructions on the IB website.
Paper Trading
IB supports paper trading. Follow the Opening a Paper Trading Account page in the IB documentation to set up your paper
trading account.
If you want to use IB market data and trade with your paper trading account, follow these steps:
1. Log in to the IB Client Portal.
2. In the top-right corner, click the person icon and then click Settings .
3. In the Account Configuration section, click Paper Trading Account .
4. Click Yes .
5. Click Save .
The IB paper trading environment simulates most aspects of a production Trader Workstation account, but you may encounter
some differences due to its construction as a simulator with no execution or clearing abilities.
Insured Bank Deposit Sweep Program
LEAN doesn't support IB accounts in the Insured Bank Deposit Sweep Program because when LEAN reads your account
balances, it includes cash that's in the FDIC Sweep Account Cash, which isn't tradable. For example, if your account has
150KUSDofcash, only100K may be available to trade if $50K is in FDIC Sweep Account Cash.
Dividend Election
Dividend election is an option where you can elect how you wish to receive your dividends for stocks and mutual funds. You
must turn automatic dividend election off to receive them in cash. Reinvestment can change the quantity of shares you own to
fractional shares, and LEAN doesn't support fractional trading . For example, if your account has 1270.8604 shares of TLT after
dividend reinvestment, you cannot liquidate the position.
Asset Classes
Our Interactive Brokers integration supports the following asset classes:
US Equities
Equity Options
Forex
Futures
Future Options
Indices
Index Options
CFD
You may not be able to trade all assets with IB. For example, if you live in the EU, you can't trade US ETFs. You can trade the
CFD equivalent . Check with your local regulators to know which assets you are allowed to trade. You may need to adjust
settings in your brokerage account to live trade some assets.
Data Providers
You might need to purchase an IB market data subscription for your trading. For more information about live data providers, see
Datasets .
Orders
We model the IB API by supporting several order types, order properties, and order updates. When you deploy live algorithms,
you can place manual orders through the IDE.
Order Types
The following table describes the order types that our IB integration supports: supports. For specific details about each order
type, refer to the IB documentation.
Order Type IB Documentation Page
Market Market Orders
Limit Limit Orders
Limit if touched Limit if Touched Orders
Stop market Stop Orders
Stop limit Stop-Limit Orders
Trailing stop Trailing Stop Orders
Market on open Market-on-Open (MOO) Orders
Market on close Market-on-Close (MOC) Orders
Combo market Spread Orders
Combo limit Spread Orders
Combo leg limit Spread Orders
Exercise Option Options Exercise
The following table describes the available order types for each asset class that IB supports:
Order Type US Equity
Equity
Options
Forex Futures
Futures
Options
Index
Options
CFD
Market
Limit
Limit if
touched
Stop
market
Stopl imit
Trailing
stop
Market on
open
Market on
close
Combo
market
Combo limit
Combo leg
limit
Exercise
Option
Not
supported
for cashsettled
Options
Order Properties
We model custom order properties from the IB API. The following table describes the members of the
InteractiveBrokersOrderProperties object that you can set to customize order execution. The table does not include the
methods for FA accounts .
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
outside_regular_trading
_hours
bool
A flag to signal that the
order may be triggered and
filled outside of regular
trading hours.
False
Updates
We model the IB API by supporting order updates .
Financial Advisor Group Orders
To place FA group orders, see Financial Advisors .
Fractional Trading
The IB API and FIX/CTCI don't support fractional trading .
Handling Splits
If you're using raw data normalization and you have active orders with a limit, stop, or trigger price in the market for a US Equity
when a stock split occurs, the following properties of your orders automatically adjust to reflect the stock split:
Quantity
Limit price
Stop price
Trigger price
Fill Time
IB has a 400 millisecond fill time for live orders.
Brokerage Liquidations
When IB liquidates part of your position, you receive an order event that contains the Brokerage Liquidation message.
Fees
To view the IB trading fees, see the Commissions page on the IB website. To view how we model their fees, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you have more than
$25,000 in your brokerage account, you can use the PatternDayTradingMarginModel to make use of the 4x intraday leverage
and 2x overnight leverage available on most brokerages from the PDT rule .
Slippage
Orders through IB do not experience slippage in backtests. In IB paper trading and live trading, your orders may experience
slippage.
To view how we model IB slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests. In IB paper trading and live trading, if the quantity of your market
orders exceeds the quantity available at the top of the order book, your orders are filled according to what is available in the
order book.
To view how we model IB order fills, see Fills .
Settlements
If you trade with a margin account, trades settle immediately
To view how we model settlement for IB trades, see Settlement .
Security and Stability
Note the following security and stability aspects of our IB integration.
Account Credentials
When you deploy live algorithms with IB , we don't save your brokerage account credentials.
API Outages
We call the IB API to place live trades. Sometimes the API may be down. Check the IB status page to see if the API is currently
working.
Connections
By default, IB only supports one connection at a time to your account. If you interfere with your brokerage account while an
algorithm is connected to it, the algorithm may stop executing. If you deploy a live running algorithm with your IB account and
want to open Trader Workstation (TWS) with the same IB account, create a second user on your IB account and log in to TWS
with the new user credentials. To run more than one algorithm with IB, open an IB subaccount for each additional algorithm.
If you can't log in to TWS with your credentials, contact IB. If you can log in to TWS but can't log in to the deployment wizard,
contact us and provide the algorithm ID and deployment ID.
SMS 2FA
Our IB integration doesn't support Two-Factor Authentication (2FA) via SMS or the Online Security Code Card. Use the IB Key
Security via IBKR Mobile instead.
System Resets
You'll receive a notification on your IB Key device every Sunday to re-authenticate the connection between IB and your live
algorithm. When you deploy your algorithm , you can select a time on Sunday to receive the notification. If you don't reauthenticate before the timeout period, your algorithm quits executing. Ensure your IB Key device has sufficient battery for the
time you expect to receive the notification. If you don't receive a notification, see I am not receiving IBKR Mobile notifications on
the IB website.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the IB brokerage:
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Interactive Brokers from the drop-down menu.
4. Enter your IB user name, ID, and password.
if ($localPlatform) { include(DOCS_RESOURCES."/brokerages/interactive-brokers/paper-trading-data-feeds.html"); } ?>
Your account details are not saved on QuantConnect.
5. In the Weekly Restart UTC field, enter the Coordinated Universal Time (UTC) time of when you want to receive notifications
on Sundays to re-authenticate your account connection.
For example, 4 PM UTC is equivalent to 11 AM Eastern Standard Time, 12 PM Eastern Daylight Time, 8 AM Pacific Standard
Time, and 9 AM Pacific Daylight Time. To convert from UTC to a different time zone, see the UTC Time Zone Converter on
the UTC Time website.
If your IB account has 2FA enabled, you receive a notification on your IB Key device every Sunday to re-authenticate the
connection between IB and your live algorithm. If you don't re-authenticate before the timeout period, your algorithm quits
executing.
 Charts  Statistics  Code Clone Algorithm
6. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
7. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
In most cases, we suggest using both the QC and IB data providers .
If you use IB data provider and trade with a paper trading account, you need to share the market data subscription with
your paper trading account. For instructions on sharing market data subscription, see Account Types .
8. (Optional) Set up notifications .
9. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
10. Click Deploy .
11. If your IB account has 2FA enabled, tap the notification on your IB Key device and then enter your pin.
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Troubleshooting
The following table describes errors you may see when deploying to IB:
Error Message(s) Possible Cause and Fix
Login failed.
The credentials you provided are incorrect. Typically, the
password contains leading and/or trailing white spaces.
Copy the password to a text editor to ensure the password
is correct. If you can't log in to Trader Workstation (TWS)
with your credentials, contact IB. If you can log in to TWS
but can't log in to the deployment wizard, contact us and
provide the algorithm Id and deployment Id.
Login to the IB Gateway failed because
a user account-tasks is required.
Download IB Gateway , run it, and follow the instructions
provided.
An existing session was detected and will not be
automatically disconnected.
Historical Market Data Service error message: Trading TWS
session is connected from a different IP address.
IB still recognizes your previous live deployment as being
partially connected. It can take a minute to fully disconnect.
For more information, see Security and Stability >
Connections .
The two factor authentication request timed out.
A security dialog was detected for Code Card
Authentication.
Unknown message window detected: Challenge: 123 456
You haven't replied to the two factor authentication
requests. The code card authentication ("Challenge") is
triggered when you don't reply to the IB mobile 2FA
requests. Ensure your IB Key device has sufficient battery
for the time you expect to receive the notification. If you
don't receive a notification, see I am not receiving IBKR
Mobile notifications on the IB website.
API support is not available for accounts that support free
trading. Upgrade your plan from IBKR Lite to IBKR Pro.
No security definition has been found for the request.
Your algorithm added an invalid or unsupported security.
For example, a delisted stock, an expired contract,
inexistent contract (invalid expiration date or strike price),
or a warrant (unsupported). If the security should be valid
and supported , open a support ticket and attach the live
deployment Id. The algorithm will continue running, but it
won't trade the security. If you don't want to deploy to an
account with an invalid or unsupported security, set
self.settings.ignore_unknown_assets is False .
Requested market data is not subscribed.
Historical Market Data Service error message: No market
data permissions for ...
Your algorithm uses the Interactive Brokers Data Provider ,
but you don't have a subscription to it. Subscribe to the
data bundle you need , contact IB, or re-deploy the
algorithm with a different data provider. Try the
QuantConnect or the hybrid QuantConnect + Interactive
Brokers data providers on QuantConnect Cloud or try a
third-party provider.
Timeout waiting for brokerage response for brokerage
order id 37 lean id 31
IB didn't respond to an order request. Stop and re-deploy
the algorithm. On the next deployment, LEAN retrieves this
order or the positions it opened or closed.
To view the description of less common errors, see Error Codes in the TWS API Documentation. If you need further support,
open a new support ticker and add the live deployment with the error.
Live Trading > Brokerages > TradeStation
Brokerages
TradeStation
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
TradeStation was founded by brothers William (Bill) and Rafael (Ralph) Cruz in 1982 as Omega Research, Inc. In 2001, the
company converted itself from a trading software company to an online securities brokerage and renamed itself "TradeStation"
with the mission to "create the ultimate trading experience". TradeStation provides access to trading Equities, Equity Options,
and Futures for clients in over 150 markets, 34 countries, and 27 currencies. TradeStation also delivers custody, clearing,
execution, and billing on behalf of registered advisors.
To view the implementation of the TradeStation brokerage integration, see the Lean.Brokerages.TradeStation repository .
Account Types
TradeStation supports cash and margin accounts. To set the account type in an algorithm, see the TradeStation brokerage
model documentation .
Create an Account
To create a TradeStation account, follow the account creation wizard on the TradeStation website.
Paper Trading
TradeStation supports paper trading through their trading simulator after you fund your brokerage account.
Asset Classes
Our TradeStation integration supports trading the following asset classes:
US Equities
Equity Options
Futures
Index
Index Options
You may not be able to trade all assets with TradeStation . For example, if you live in the EU, you can't trade US ETFs. Check
with your local regulators to know which assets you are allowed to trade. You may need to adjust settings in your brokerage
account to live trade some assets.
Data Providers
You might need to purchase a TradeStation market data subscription for your trading. For more information about live data
providers, see Datasets .
Orders
We model the TradeStation API by supporting several order types, order properties, and order updates. When you deploy live
algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our TradeStation integration supports:
Order Type Equity Equity Options Index Options Futures
Market
Limit
Stop market
Stop limit
Trailing stop
Market on Open
Market on Close
Combo market
Combo limit
Order Properties
We model the TradeStation API. The following table describes the members of the TradeStationOrderProperties object that
you can set to customize order execution.
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
outside_regular_trading
_hours
bool
A flag to signal that the
order may be triggered and
filled outside of regular
trading hours.
False
post_only bool
This flag will ensure the
limit order executes only as
a maker (no fee) order. If
part of the order results in
taking liquidity rather than
providing, it will be rejected
and no part of the order will
execute. Equities only.
False
Updates
We model the TradeStation API by supporting order updates .
Handling Splits
If you're using raw data normalization and you have active orders with a limit, stop, or trigger price in the market for a US Equity
when a stock split occurs, the following properties of your orders automatically adjust to reflect the stock split:
Quantity
Limit price
Stop price
Trigger price
Fees
To view the TradeStation trading fees, see the Pricing page on the TradeStation website. To view how we model their fees, see
Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you have more than
$25,000 in your brokerage account, you can use the PatternDayTradingMarginModel to make use of the 4x intraday leverage
and 2x overnight leverage available on most brokerages from the PDT rule .
Slippage
Orders through TradeStation do not experience slippage in backtests and QuantConnect Paper Trading . In live trading, your
orders may experience slippage.
To view how we model TradeStation slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model TradeStation order fills, see Fills .
Settlements
If you trade with a margin account, trades settle immediately
To view how we model settlement for TradeStation trades, see Settlement .
Security and Stability
When you deploy live algorithms with TradeStation , we don't save your brokerage account credentials.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the TradeStation brokerage:
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click TradeStation from the drop-down menu.
4. Click on the Environment field and then click one of the environments.
The following table shows the supported environments:
Environment Description
Simulator Trade with paper money
Live Trade with real money
5. Click on Authenticate button.
6. On the TradeStation website, login to your account to grant QuantConnect access to your account information and
authorization.
7. Click on the Select Account Id field and then click on one of your accounts.
8. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
9. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
In most cases, we suggest using the QuantConnect data provider , the TradeStation data provider , or both. The order you
set them in the deployment wizard defines their order of precedence in Lean.
10. (Optional) Set up notifications .
11. Configure the Automatically restart algorithm setting.
 Charts  Statistics  Code Clone Algorithm
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
12. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Alpaca
Brokerages
Alpaca
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Alpaca was founded by Yoshi Yokokawa and Hitoshi Harada in 2015 as a database and machine learning company. In 2018,
Alpaca Securities LLC (Alpaca Securities) became a registered US broker-dealer with the Financial Industry Regulatory Authority
( FINRA ) with the mission to "open financial services to everyone on the planet". In 2022, Alpaca Crypto LLC (Alpaca Crypto)
became a registered Money Services Business (MSB) with the Financial Crimes Enforcement Network ( FinCEN ). Alpaca
provides access to trading Equities, Options, and Crypto for clients in over 30 countries. Alpaca also delivers custody, clearing,
execution, and billing on behalf of registered advisors.
To view the implementation of the Alpaca brokerage integration, see the Lean.Brokerages.Alpaca repository .
Account Types
Alpaca supports cash and margin accounts. To set the account type in an algorithm, see the Alpaca brokerage model
documentation .
Create an Account
Follow the account creation wizard on the Alpaca website to create an Alpaca account.
You will need API credentials to deploy live algorithms with your brokerage account. After you have an account, open the
Dashboard page, click the button in the top left to select your real money trading account, and then click Generate New Keys .
Store your API Key and Secret somewhere safe.
Paper Trading
Alpaca supports paper trading. When you create an Alpaca account, your account is a paper trading account by default. To get
your paper trading API credentials, open the Dashboard page, click the button in the top left to select your paper trading
account, and then click Generate New Keys . For more information about their paper trading environment, see their rules and
assumptions documentation.
Asset Classes
Our Alpaca integration supports the following asset classes:
US Equities
Equity Options
Crypto
You may not be able to trade all assets with Alpaca. For example, if you live in the EU, you can't trade US ETFs. Check with your
local regulators to know which assets you are allowed to trade. You may need to adjust settings in your brokerage account to
live trade some assets.
Data Providers
You might need to purchase a Alpaca market data subscription for your trading. For more information about live data providers,
see Datasets .
Orders
We model the Alpaca API by supporting several order types, order properties, and order updates. When you deploy live
algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Alpaca integration supports:
Order Type Equity Equity Options Crypto
Market
Limit
Stop market
Stop limit
Market on Open
Market on Close
Order Properties
We model the Alpaca API. The following table describes the members of the AlpacaOrderProperties object that you can set to
customize order execution.
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
outside_regular_trading
_hours
bool
A flag to signal that the
order may be triggered and
filled outside of regular
trading hours.
False
Updates
We model the Alpaca API by supporting order updates .
Handling Splits
If you're using raw data normalization and you have active orders with a limit, stop, or trigger price in the market for a US Equity
when a stock split occurs, the following properties of your orders automatically adjust to reflect the stock split:
Quantity
Limit price
Stop price
Trigger price
Fees
The Alpaca trading for Equity and Equity Options is commission-free. To view the Alpaca trading fees for Crypto, see the Crypto
Fees page on the Alpaca website. To view how we model their fees, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you have more than
$25,000 in your brokerage account, you can use the PatternDayTradingMarginModel to make use of the 4x intraday leverage
and 2x overnight leverage available on most brokerages from the PDT rule .
Slippage
Orders through Alpaca do not experience slippage in backtests and QuantConnect Paper Trading . In live trading, your orders
may experience slippage.
To view how we model Alpaca slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model Alpaca order fills, see Fills .
Settlements
If you trade with a margin account, trades settle immediately
To view how we model settlement for Alpaca trades, see Settlement .
Security and Stability
Note the following security and stability aspects of our Alpaca integration.
Account Credentials
When you deploy live algorithms with Alpaca, we don't save your brokerage account credentials.
API Outages
We call the Alpaca API to place live trades. Sometimes the API may be down. Check the Alpaca status page to see if the API is
currently working.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Alpaca brokerage:
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Alpaca from the drop-down menu.
4. Click on the Environment field and then click one of the environments.
The following table shows the supported environments:
Environment Description
Paper Trade with paper money
Live Trade with real money
5. Check the Authorization check box and then click Authenticate .
6. On the Alpaca website, click Allow to grant QuantConnect access to your account information and authorization.
7. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
8. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
In most cases, we suggest using the QuantConnect data provider , the Alpaca data provider , or both. The order you set
them in the deployment wizard defines their order of precedence in Lean.
If you add the Alpaca data provider, enter your API key and secret. To get your API key and secret, see Account Types .
Your account details are not saved on QuantConnect.
9. (Optional) Set up notifications .
10. Configure the Automatically restart algorithm setting.
 Charts  Statistics  Code Clone Algorithm
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
11. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Charles Schwab
Brokerages
Charles Schwab
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
The Charles Schwab Corporation was founded by Charles R. Schwab in 1971. Charles Schwab provides access to trading
Equities, Options, Index Options, and other assets for clients with no account or trade minimums, or hidden fees .
Account Types
Charles Schwab supports cash and margin accounts. To set the account type in an algorithm, see the Charles Schwab
brokerage model documentation .
Create an Account
Follow the account creation wizard on the Charles Schwab website to create a Charles Schwab account.
Paper Trading
The Charles Schwab API doesn't support paper trading, but you can follow these steps to simulate it with QuantConnect:
1. In the initialize method of your algorithm, set the Charles Schwab brokerage model and your account type .
2. Deploy your algorithm with the QuantConnect Paper Trading brokerage .
Asset Classes
Our Charles Schwab integration supports the following asset classes:
US Equities
Equity Options
Index
Index Options
You may not be able to trade all assets with Charles Schwab. For example, if you live in the EU, you can't trade US ETFs. Check
with your local regulators to know which assets you are allowed to trade. You may need to adjust settings in your brokerage
account to live trade some assets.
Data Providers
You might need to purchase a Charles Schwab market data subscription for your trading. For more information about live data
providers, see Datasets .
Orders
We model the Charles Schwab API by supporting several order types, the TimeInForce order property, and order updates.
When you deploy live algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Charles Schwab integration supports:
Order Type Equity Equity Options Index Options
Market
Limit
Stop market
Market on open
Market on close
Combo market
Combo limit
Order Properties
We model custom order properties from the Charles Schwab API. The following table describes the members of the
CharlesSchwabOrderProperties object that you can set to customize order execution.
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
extended_regular_tradin
g_hours
bool
If set to true, allows orders
to also trigger or fill outside
of regular trading hours.
False
Updates
We model the Charles Schwab API by supporting order updates .
Handling Splits
If you're using raw data normalization and you have active orders with a limit, stop, or trigger price in the market for a US Equity
when a stock split occurs, the following properties of your orders automatically adjust to reflect the stock split:
Quantity
Limit price
Stop price
Trigger price
Fees
The Charles Schwab trading for Equity is commission-free. To view the Charles Schwab trading fees, see the Pricing page on
the Charles Schwab website. To view how we model their fees, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you have more than
$25,000 in your brokerage account, you can use the PatternDayTradingMarginModel to make use of the 4x intraday leverage
and 2x overnight leverage available on most brokerages from the PDT rule .
Slippage
Orders through Charles Schwab do not experience slippage in backtests and QuantConnect Paper Trading . In live trading, your
orders may experience slippage.
To view how we model Charles Schwab slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model Charles Schwab order fills, see Fills .
Settlements
If you trade with a margin account, trades settle immediately
To view how we model settlement for Charles Schwab trades, see Settlement .
Security and Stability
When you deploy live algorithms with Charles Schwab, we don't save your brokerage account credentials.
Charles Schwab only supports authenticating one account at a time per user. If you have an algorithm running with Charles
Schwab and then deploy a second one, the first algorithm stops running.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Funds are available for API trading 24 hours after the deposit.
Demo Algorithm
The following algorithm demonstrates the functionality of the Charles Schwab brokerage:
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Charles Schwab from the drop-down menu.
4. Check the Authorization check box and then click Authenticate .
5. On the Charles Schwab website, log in, and select your Schwab accounts to link. Click Allow to grant QuantConnect
access to your account information and authorization.
6. Click on the Select Account Id field and then click on one of your accounts.
7. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
8. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
In most cases, we suggest using the QuantConnect data provider , the Charles Schwab data provider , or both. The order
you set them in the deployment wizard defines their order of precedence in Lean.
9. (Optional) Set up notifications .
10. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
11. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Troubleshooting
 Charts  Statistics  Code Clone Algorithm
The following table describes errors you may see when deploying to Charles Schwab:
Error Message(s) Possible Cause and Fix
Name = Account_Sys_0005,Description = No trades are currently allowed
Your account may not have the necessary permissions to
place trades. Log into schwab.com , navigate to Trade >
Trading Platforms , and click on Enable trading on
thinkorswim. The permissions may take up to 48 hours to
take effect. If the issue persists, contact Charles Schwab
support.
If you need further support, open a new support ticker and add the live deployment with the error.
Live Trading > Brokerages > Binance
Brokerages
Binance
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Binance was founded by Changpeng Zhao in 2017 with the goal to "increase the freedom of money globally". Binance provides
access to trading Crypto through spot markets and perpetual Futures. They serve clients with no minimum deposit when
depositing Crypto. Binance also provides an NFT marketplace, a mining pool, and services to deposit Crypto coins in liquidity
pools to earn rewards.
To view the implementation of the Binance brokerage integration, see the Lean.Brokerages.Binance repository .
Account Types
Binance supports cash and margin accounts for spot trades, but only supports margin accounts for Futures trades. Binance US
only supports cash accounts. To set the account type in an algorithm, see the Binance brokerage model documentation .
Create an Account
Follow the account creation wizard on the Binance.com or Binance.us website to create a Binance account.
You will need API credentials to deploy live algorithms with your brokerage account. After you open your account, create API
credentials and store them somewhere safe. As you create credentials, make the following configurations:
Select the Restrict access to trusted IPs only check box and then enter our IP address, 146.59.85.21.
If you are going to trade Crypto Futures, select the Enable Futures check box.
Paper Trading
Binance supports paper trading through the Binance Spot Test Network. You don't need a Binance account to create API
credentials for the Spot Test Network.
Follow these steps to set up paper trading with the Binance Spot Test Network:
1. Log in to the Binance Spot Test Network with your GitHub credentials.
2. In the API Keys section, click Generate HMAC_SHA256 Key .
3. Enter a description and then click Generate .
4. Store your API key and API key secret somewhere safe.
Paper trading Binance Crypto Futures or with Binance US isn't currently available.
Sub-Accounts
Our Binance and Binance US integrations don't support trading with sub-accounts. You must use your main account.
Asset Classes
Our Binance integration supports trading Crypto and Crypto Futures .
Our Binance US integration supports trading Crypto .
Data Providers
The QuantConnect data provider provides Crypto data during live trading.
Orders
We model the Binance and Binance US APIs by supporting several order types, supporting order properties, and not supporting
order updates. When you deploy live algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Binance and Binance US integrations
support:
Order Type Crypto Crypto Futures
Market
Limit
Stop Market
Stop limit
Order Properties
We model custom order properties from the Binance and Binance US APIs. The following table describes the members of the
BinanceOrderProperties object that you can set to customize order execution:
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
post_only bool
A flag to signal that the
order must only add
liquidity to the order book
and not take liquidity from
the order book. If part of
the order results in taking
liquidity rather than
providing liquidity, the
order is rejected without
any part of it being filled.
Updates
We model the Binance and Binance US APIs by not supporting order updates, but you can cancel an existing order and then
create a new order with the desired arguments. For more information about this workaround, see the Workaround for
Brokerages That Donʼt Support Updates .
Fees
To view the Binance or Binance US trading fees, see the Trading Fees page on the Binance.com website or the Fee Structure
page on the Binance.us website. To view how we model their fees, see Fees . The Binance Spot Test Network does not charge
order fees.
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you trade Crypto
Perpetual Futures, we model the margin cost and payments of your Crypto Future holdings by directly adjusting your portfolio
cash. For more information about Futures margin interest modeling, see the Binance Futures Model .
Slippage
Orders through Binance and Binance US do not experience slippage in backtests and QuantConnect Paper Trading . In live
trading, your orders may experience slippage.
To view how we model Binance and Binance US slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model Binance and Binance US order fills, see Fills .
Settlements
Trades settle immediately after the transaction
To view how we model settlement for Binance and Binance US trades, see Settlement .
Security and Stability
When you deploy live algorithms with Binance or Binance US, we don't save your brokerage account credentials.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Binance and Binance US brokerages:
Binance
Binance US
Virtual Pairs
All fiat and Crypto currencies are individual assets. When you buy a pair like BTCUSD, you trade USD for BTC. In this case, LEAN
removes some USD from your portfolio cashbook and adds some BTC. The virtual pair BTCUSD represents your position in that
trade, but the virtual pair doesn't actually exist. It simply represents an open trade. When you deploy a live algorithm, LEAN
populates your cashbook with the quantity of each currency, but it can't get your position of each virtual pair.
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
 Charts  Statistics  Code Clone Algorithm
 Charts  Statistics  Code Clone Algorithm
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Binance Exchange from the drop-down menu.
4. Enter your API key and secret.
To generate your API credentials, see Account Types . Your account details are not saved on QuantConnect.
5. Click on the Environment field and then click one of the environments.
The following table shows the supported environments:
Environment Description
Real Trade with real money
Demo
Trade with paper money through the Binance Global
brokerage
6. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
7. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
8. If your brokerage account has existing cash holdings, follow these steps ( see video ):
1. In the Algorithm Cash State section, click Show .
2. Click Add Currency .
3. Enter the currency ticker (for example, USD or CAD) and a quantity.
9. If your brokerage account has existing position holdings, follow these steps ( see video ):
1. In the Algorithm Holdings State section, click Show .
2. Click Add Holding .
3. Enter the symbol ID, symbol, quantity, and average price.
10. (Optional) Set up notifications .
11. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
12. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > ByBit
Brokerages
ByBit
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Bybit was co-founded by Ben Zhou in March 2018 with the goal to offer a professional platform where Crypto traders can find an
ultra-fast matching engine, excellent customer service, and multilingual community support. Bybit provides access to trading
Crypto and Crypto Futures for clients outside of excluded jurisdictions with low minimum deposits to set up an account. For
more information about Crypto and fiat deposits , see the Bybit documentation. Bybit also provides Crypto staking, initial DEX
offerings, and community airdrops.
To view the implementation of the Bybit brokerage integration, see the Lean.Brokerages.Bybit repository .
Account Types
Bybit supports cash and margin accounts. To set the account type in an algorithm, see the Bybit brokerage model
documentation .
Create an Account
Follow the How to Register an Account tutorial on the Bybit website to create a Bybit account.
You will need API credentials to deploy live algorithms. After you have an account, create API credentials and store them
somewhere safe. As you create credentials, make the following configurations:
Under API Key Usage, select API Transaction .
Choose a name for the API key.
Under API Key Permissions, select Read-Write .
Select the Only IPs with permissions granted are allowed to access the OpenAPI check box and then enter our IP address,
146.59.85.21.
Select the Unified Trading check box.
Unified trading enables read-write permissions for the following queries:
Order
Positions
Trade
Paper Trading
Our integration doesn't support paper trading through the Bybit Demo Trading environment, but you can follow these steps to
simulate it with QuantConnect:
1. In the initialize method of your algorithm, set the Bybit brokerage model.
2. Deploy your algorithm with the QuantConnect Paper Trading brokerage .
Asset Classes
Our Bybit integration supports trading Crypto and Crypto Futures .
Data Providers
The QuantConnect data provider provides Crypto data during live trading.
Orders
We model the Bybit API by supporting several order types, order properties, and order updates. When you deploy live
algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Bybit integration supports:
Order Type Crypto Crypto Futures
Market
Limit
Stop market
Stop limit
Order Properties
We model custom order properties from the Bybit API. The following table describes the members of the BybitBrokerageModel
object that you can set to customize order execution:
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
post_only bool
A flag to signal that the
order must only add
liquidity to the order book
and not take liquidity from
the order book. If part of
the order results in taking
liquidity rather than
providing liquidity, the
order is rejected without
any part of it being filled.
This order property is only
available for limit orders.
reduce_only bool/NoneType
A flag to signal that the
order must only reduce
your current position size.
For more information about
this order property, see
Reduce-Only Order on the
Bybit website.
Updates
We model the Bybit API by supporting order updates for Crypto Future assets that have one of the following order states :
OrderStatus.NEW
OrderStatus.PARTIALLY_FILLED
OrderStatus.SUBMITTED
OrderStatus.FILLED
In cases where you can't update an order, you can cancel the existing order and then create a new order with the desired
arguments. For more information about this workaround, see the Workaround for Brokerages That Donʼt Support Updates .
Fees
To view the Bybit trading fees, see the Trading Fees Schedule page on the Bybit website. To view how we model their fees, see
Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements.
Slippage
Orders through Bybit do not experience slippage in backtests. In Bybit paper trading and live trading, your orders may
experience slippage.
To view how we model Bybit slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests. In Bybit paper trading and live trading, if the quantity of your
market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is available in
the order book.
To view how we model Bybit order fills, see Fills .
Settlements
Trades settle immediately after the transaction
To view how we model settlement for Bybit trades, see Settlement .
Security and Stability
When you deploy live algorithms with Bybit, we don't save your brokerage account credentials.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Bybit brokerage:
Virtual Pairs
All fiat and Crypto currencies are individual assets. When you buy a pair like BTCUSD, you trade USD for BTC. In this case, LEAN
removes some USD from your portfolio cashbook and adds some BTC. The virtual pair BTCUSD represents your position in that
trade, but the virtual pair doesn't actually exist. It simply represents an open trade. When you deploy a live algorithm, LEAN
populates your cashbook with the quantity of each currency, but it can't get your position of each virtual pair.
 Charts  Statistics  Code Clone Algorithm
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Bybit Exchange from the drop-down menu.
4. Enter your API key and secret.
To generate your API credentials, see Account Types . Your account details are not saved on QuantConnect.
5. Click the VIP Level field and then click your level from the drop-down menu.
For more information about VIP levels, see FAQ — Bybit VIP Program on the Bybit website.
6. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
7. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
8. If your brokerage account has existing position holdings, follow these steps ( see video ):
1. In the Algorithm Holdings State section, click Show .
2. Click Add Holding .
3. Enter the symbol ID, symbol, quantity, and average price.
9. (Optional) Set up notifications .
10. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
11. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Troubleshooting
The following table describes errors you may see when deploying to Bybit:
Error Message(s) Possible Cause and Fix
Invalid API-key, IP, or permissions for action.
The credentials you provided are incorrect. Typically, the
API key and API Secret contains leading and/or trailing
white spaces. Copy the API key and API Secret to a text
editor to ensure their are correct.
Your API keys do not have read-write permissions. For
more information, see Account Types .
You haven't enabled Cross Margin Trading yet.
Your account uses isolated margin. To enable cross margin,
please head to the PC trading site or the Bybit app. For more
information, see Isolated Margin/Cross Margin .
If you need further support, open a new support ticker and add the live deployment with the error.
Live Trading > Brokerages > Tradier
Brokerages
Tradier
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Tradier was founded by Dan Raju, Peter Laptewicz, Jason Barry, Jeyashree Chidambaram, and Steve Agalloco in 2012 with the
goal to "deliver a choice of low-cost, high-value brokerage services to traders". Tradier provides access to trading Equities and
Options for clients in over 250 countries and territories with no minimum deposit for cash accounts . Tradier also delivers
custody, clearing, execution, and billing on behalf of registered advisors.
To view the implementation of the Tradier brokerage integration, see the Lean.Brokerages.Tradier repository .
Account Types
Tradier supports cash and margin accounts. To set the account type in an algorithm, see the Tradier brokerage model
documentation .
Create an Account
Follow the account creation wizard on the Tradier website to create a Tradier account.
You will need your account ID and access token to deploy live algorithms. After you have an account, get your account ID and
token from the Settings > API Access page on the Tradier website. Your account ID is the alpha-numeric code in a drop-down
field on the page.
Paper Trading
Tradier supports paper trading, but with the following caveats:
Account activity is unavailable since this information is populated from Tradier's clearing firm.
Streaming Tradier market data is unavailable due to exchange restrictions related to delayed data.
To get your paper trading account number and access token, open the API Access page on the Tradier website and then scroll
down to the Sandbox Account Access (Paper Trading) section.
If you trade Equities, you can use the QuantConnect data provider to get real-time data. If you trade Options, you must use
delayed data from the Tradier data provider . If you trade Equities and Options, use both data providers. However, if you trade
with the demo environment, Tradier doesn't offer streaming market data due to exchange restrictions related to delayed data,
so you must use our data provider.
Asset Classes
Our Tradier integration supports trading US Equities and Equity Options .
You may not be able to trade all assets with Tradier. For example, if you live in the EU, you can't trade US ETFs. Check with your
local regulators to know which assets you are allowed to trade. You may need to adjust settings in your brokerage account to
live trade some assets.
Data Providers
You might need to purchase a Tradier market data subscription for your trading. For more information about live data providers,
see Datasets .
Orders
We model the Tradier API by supporting several order types and the TimeInForce order property. Tradier partially supports
order updates, but does not support trading during extended market hours. When you deploy live algorithms, you can place
manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Tradier integration supports:
Order Type Equity Equity Options
Market
Limit
Stop market
Stop limit
Order Properties
We model the Tradier API. The following table describes the members of the TradierOrderProperties object that you can set
to customize order execution.
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
outside_regular_trading
_hours
bool
A flag to signal that the
order may be triggered and
filled outside of regular
trading hours.
False
Updates
We model the Tradier API by supporting most order updates . To update the quantity of an order, cancel the order and then
submit a new order with the desired quantity. For more information about this workaround, see the Workaround for Brokerages
That Donʼt Support Updates .
Extended Market Hours
Tradier doesn't support extended market hours trading. If you place an order outside of regular trading hours, the order will be
processed at market open.
Automatic Cancellations
If you have open orders for a security when it performs a reverse split, Tradier automatically cancels your orders.
Errors
To view the order-related error codes from Tradier, see Error Responses in their documentation.
Fees
To view the Tradier trading fees, see the Pricing page on the Tradier website. To view how we model their fees, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you have more than
$25,000 in your brokerage account, you can use the PatternDayTradingMarginModel to make use of the 4x intraday leverage
and 2x overnight leverage available on most brokerages from the PDT rule .
Slippage
Orders through Tradier do not experience slippage in backtests and QuantConnect Paper Trading . In live trading, your orders
may experience slippage.
To view how we model Tradier slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model Tradier order fills, see Fills .
Settlements
If you trade with a margin account, trades settle immediately
To view how we model settlement for Tradier trades, see Settlement .
Security and Stability
Note the following security and stability aspects of our Tradier integration.
Account Credentials
When you deploy live algorithms with Tradier, we don't save your brokerage account credentials.
API Outages
We call the Tradier API to place live trades. Sometimes the API may be down. Check the Tradier status page to see if the API is
currently working.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Tradier brokerage:
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Tradier from the drop-down menu.
4. Enter your Tradier account Id and token.
To get your account ID and token, see the Create an Account section in the Account Types documentation. Your account
details are not saved on QuantConnect.
5. Click the Environment field and then click one of the environments from the drop-down menu.
The following table shows the supported environments:
Environment Description
Real Trade with real money
Demo Trade with paper money
6. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
7. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
In most cases, we suggest using the QuantConnect data provider , the Tradier data provider , or both. The order you set
them in the deployment wizard defines their order of precedence in Lean. In the demo environment, Tradier doesn't offer
streaming market data due to exchange restrictions related to delayed data.
8. (Optional) Set up notifications .
9. Configure the Automatically restart algorithm setting.
 Charts  Statistics  Code Clone Algorithm
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
10. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Kraken
Brokerages
Kraken
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Kraken was founded by Jesse Powell in 2011 with the goal to "accelerate the adoption of cryptocurrency so that you and the
rest of the world can achieve financial freedom and inclusion". Kraken provides access to trading Crypto through spot and
Futures markets for clients with a minimum deposit of around 0 − 150 USD for currency and Crypto deposits . Kraken also
provides staking services, educational content, and a developer grant program.
To view the implementation of the Kraken brokerage integration, see the Lean.Brokerages.Kraken repository .
Account Types
Kraken supports cash and margin accounts. To set the account type in an algorithm, see the Kraken brokerage model
documentation .
Create an Account
Follow the account creation wizard on the Kraken website to create a Kraken account.
You will need API credentials to deploy live algorithms with your brokerage account. After you open your account, create API
credentials and store them somewhere safe.
Paper Trading
The Kraken brokerage doesn't support paper trading, but you can follow these steps to simulate it with QuantConnect:
1. In the initialize method of your algorithm, set the Kraken brokerage model and your account type .
2. Deploy your algorithm with the QuantConnect Paper Trading brokerage .
Rewards Program
QuantConnect doesn't account for assets you enable in Kraken's rewards program .
Asset Classes
Our Kraken integration supports trading Crypto .
Data Providers
The QuantConnect data provider provides Crypto data during live trading.
Orders
We model the Kraken API by supporting several order types, supporting order properties, and not supporting order updates.
When you deploy live algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Kraken integration supports:
Order Type Crypto
Market
Limit
Limit if touched
Stop market
Stop limit
Order Properties
We model custom order properties from the Kraken API. The following table describes the members of the
KrakenOrderProperties object that you can set to customize order execution:
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
post_only bool
A flag to signal that the
order must only add
liquidity to the order book
and not take liquidity from
the order book. If part of
the order results in taking
liquidity rather than
providing liquidity, the
order is rejected without
any part of it being filled.
fee_in_base bool
A flag to signal that the
order fees should be paid in
the base currency, which is
the default behavior when
selling. This flag must be
the opposite of the
fee_in_quote flag.
fee_in_quote bool
A flag to signal that the
order fees should be paid in
the quote currency, which
is the default behavior
when buying. This flag
must be the opposite of the
fee_in_base flag.
no_market_price_protect
ion
bool
A flag to signal that no
Market Price Protection
should be used.
conditional_order Order
An Order that's submitted
when the primary order is
executed. The
conditional_order
quantity must match the
primary order quantity and
the conditional_order
direction must be the
opposite of the primary
order direction. This order
property is only available
for live algorithms.
Updates
We model the Kraken API by not supporting order updates, but you can cancel an existing order and then create a new order
with the desired arguments. For more information about this workaround, see the Workaround for Brokerages That Donʼt
Support Updates .
Fees
To view the Kraken trading fees, see the Fee Schedule page on the Kraken website. To view how we model their fees, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements.
Slippage
Orders through Kraken do not experience slippage in backtests and QuantConnect Paper Trading . In live trading, your orders
may experience slippage.
To view how we model Kraken slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model Kraken order fills, see Fills .
Settlements
Trades settle immediately after the transaction
To view how we model settlement for Kraken trades, see Settlement .
Security and Stability
Note the following security and stability aspects of our Kraken integration.
Account Credentials
When you deploy live algorithms with Kraken, we don't save your brokerage account credentials.
API Outages
We call the Kraken API to place live trades. Sometimes the API may be down. Check the Kraken status page to see if the API is
currently working.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Kraken brokerage:
Virtual Pairs
All fiat and Crypto currencies are individual assets. When you buy a pair like BTCUSD, you trade USD for BTC. In this case, LEAN
removes some USD from your portfolio cashbook and adds some BTC. The virtual pair BTCUSD represents your position in that
trade, but the virtual pair doesn't actually exist. It simply represents an open trade. When you deploy a live algorithm, LEAN
populates your cashbook with the quantity of each currency, but it can't get your position of each virtual pair.
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Kraken Exchange from the drop-down menu.
4. Enter your Kraken API secret and key.
Gather your API credentials from the API Management Settings page on the Kraken website. Your account details are not
saved on QuantConnect.
5. Click the Verification Tier field and then click your verification tier from the drop-down menu.
For more information about verification tiers, see Verification levels explained on the Kraken website.
6. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
7. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
8. (Optional) Set up notifications .
9. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
 Charts  Statistics  Code Clone Algorithm
10. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Coinbase
Brokerages
Coinbase
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Coinbase was founded by Brian Armstrong and Fred Ehrsam in 2012 with the goal to "increase economic freedom in the world".
Coinbase provides access to trading Crypto for clients in over 100 countries with no minimum deposit. Coinbase also provides a
self-hosted Crypto wallet, a Visa debit rewards card, and Bitcoin collateral-backed lines of credit.
To view the implementation of the Coinbase brokerage integration, see the Lean.Brokerages.Coinbase repository .
Account Types
Coinbase supports cash accounts. To set the account type in an algorithm, see the Coinbase brokerage model documentation .
Create an Account
Follow the Create a Coinbase account tutorial on the Coinbase website to create an account.
You will need API credentials to deploy live algorithms. After you have an account, see Getting Started with Advanced Trade
APIs in the Coinbase documentation to create API credentials. As you create the credentials, enable the View (read-only)
permission check box, enable the Trade (execute trades on your behalf) permission check box, and whilelist our IP address
(146.59.85.21). The following text is an example of the JSON file that contains your credentials:
organizations/2c7dhs-a3a3-4acf-aa0c-f68584f34c37/apiKeys/41090ffa-asd2-8080-815f-afaf63747e35
-----BEGIN EC PRIVATE KEY-----
\nMHcCAQEEIPcJGfXYEdLQi0iFj1xvGfPwuRNoebbwuKS4xL2NrlGWpoAoGCCqGSM49\nAwEHoUQDQgAEclN+asd/EhJ3UjOWkHmP/iqGBv5NkNJ75bUq\nVgxS4aU3/dj----END EC PRIVATE KEY-----\n
In this example, the API name is
organizations/2c7dhs-a3a3-4acf-aa0c-f68584f34c37/apiKeys/41090ffa-asd2-8080-815f-afaf63747e35
The API private key for LEAN CLI use is
-----BEGIN EC PRIVATE KEY-----
\nMHcCAQEEIPcJGfXYEdLQi0iFj1xvGfPwuRNoebbwuKS4xL2NrlGWpoAoGCCqGSM49\nAwEHoUQDQgAEclN+asd/EhJ3UjOWkHmP/iqGBv5NkNJ75bUq\nVgxS4aU3/dj----END EC PRIVATE KEY-----\n
The API private key for Cloud Platform and Local Platform use is
MHcCAQEEIPcJGfXYEdLQi0iFj1xvGfPwuRNoebbwuKS4xL2NrlGWpoAoGCCqGSM49 AwEHoUQDQgAEclN+asd/EhJ3UjOWkHmP/iqGBv5NkNJ75bUq
VgxS4aU3/djHiIuSf27QasdOFIDGJLmOn7YiQ==
Note that for the Cloud Platform and Local Platform, you need to follow these steps to adjust the API private key:
1. Remove -----BEGIN EC PRIVATE KEY-----\n .
2. Remove \n-----END EC PRIVATE KEY-----\n .
3. Replace \n with a whitespace character.
For more information about Coinbase Advanced Trade APIs, see Getting Started .
Paper Trading
The Coinbase brokerage doesn't support paper trading, but you can follow these steps to simulate it with QuantConnect:
1. In the initialize method of your algorithm, set the Coinbase brokerage model and your account type .
2. Deploy your algorithm with the QuantConnect Paper Trading brokerage .
Asset Classes
Our Coinbase integration supports trading Crypto .
Data Providers
The QuantConnect data provider provides Crypto data during live trading.
Orders
We model the Coinbase API by supporting several order types, supporting order properties, and not supporting order updates.
When you deploy live algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Coinbase integration supports:
Order Type Crypto
Market
Limit
Stop market
Stop limit
Order Properties
We model custom order properties from the Coinbase API. The following table describes the members of the
CoinbaseOrderProperties object that you can set to customize order execution:
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
TimeInForce is supported.
TimeInForce.GOOD_TIL_CA
NCELED
post_only bool
A flag that signals the order
must only add liquidity to
the order book and not take
liquidity from the order
book. If part of the order
results in taking liquidity
rather than providing
liquidity, the order is
rejected without any part of
it being filled.
self_trade_prevention_i
d
bool
A flag that signals selftrade prevention is enabled
for this order. Self-trade
prevention helps prevent an
order from crossing against
the same user, reducing the
risk of unintentional trades
within the same account.
Updates
We model the Coinbase API by not supporting order updates, but you can cancel an existing order and then create a new order
with the desired arguments. For more information about this workaround, see the Workaround for Brokerages That Donʼt
Support Updates .
Fees
To view the Coinbase trading fees, see the What are the fees on Coinbase? page on the Coinbase website. To view how we
model their fees, see Fees .
Margin
Coinbase doesn't support margin trading.
Slippage
Orders through Coinbase do not experience slippage in backtests. In Coinbase paper trading and live trading, your orders may
experience slippage.
To view how we model Coinbase slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests. In Coinbase paper trading and live trading, if the quantity of your
market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is available in
the order book.
To view how we model Coinbase order fills, see Fills .
Settlements
Trades settle immediately after the transaction
To view how we model settlement for Coinbase trades, see Settlement .
Security and Stability
Note the following security and stability aspects of our Coinbase integration.
Account Credentials
When you deploy live algorithms with Coinbase, we don't save your brokerage account credentials.
API Outages
We call the Coinbase API to place live trades. Sometimes the API may be down. Check the Coinbase status page to see if the
API is currently working.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Coinbase brokerage:
Virtual Pairs
All fiat and Crypto currencies are individual assets. When you buy a pair like BTCUSD, you trade USD for BTC. In this case, LEAN
removes some USD from your portfolio cashbook and adds some BTC. The virtual pair BTCUSD represents your position in that
trade, but the virtual pair doesn't actually exist. It simply represents an open trade. When you deploy a live algorithm, LEAN
populates your cashbook with the quantity of each currency, but it can't get your position of each virtual pair.
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
 Charts  Statistics  Code Clone Algorithm
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Coinbase from the drop-down menu.
4. Enter your Coinbase API Name and API Private Key.
To generate your API credentials, see the Create an Account section in the Account Types documentation. Your account
details are not saved on QuantConnect.
5. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
6. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
7. If your brokerage account has existing position holdings, follow these steps ( see video ):
1. In the Algorithm Holdings State section, click Show .
2. Click Add Holding .
3. Enter the symbol ID, symbol, quantity, and average price.
8. (Optional) Set up notifications .
9. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
10. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Bitfinex
Brokerages
Bitfinex
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Bitfinex was founded by Giancarlo Devasini and Raphael Nicolle in 2012 with the goal to "give our users the ultimate
cryptocurrency trading experience". Bitfinex provides access to trading Crypto for clients outside prohibited jurisdictions with
no minimum deposit to set up an account. If you fund your account with fiat currency, they enforce a 10,000 minimum for USD,
EUR, and GBP. However, if you fund your account with Crypto, they do not enforce a minimum deposit. Bitfinex also provides
Crypto staking, a mobile app, and an unrealized profit leaderboard for the traders on the platform. Bitfinex has always been at
the forefront of technological innovation in digital asset trading.
To view the implementation of the Bitfinex brokerage integration, see the Lean.Brokerages.Bitfinex repository .
Account Types
Bitfinex supports cash and margin accounts. To set the account type in an algorithm, see the Bitfinex brokerage model
documentation .
Use AccountType.Cash to connect to your Exchange wallet or AccountType.Margin to connect to your Margin wallet. You can
not connect to your Funding or Capital Raise wallet. If you provide the wrong AccountType to the set_brokerage_model method,
you may connect to an empty wallet, causing Lean to throw a warning. If you have a currency in your wallet that ends with "F0",
it will not load into your CashBook .
Create an Account
Follow the account creation wizard on the Bitfinex website to create a Bitfinex account.
You will need API credentials to deploy live algorithms. After you have an account, create API credentials and store them
somewhere safe.
Paper Trading
Bitfinex supports paper trading with only the TESTBTCTESTUSD and TESTBTCTESTUSDT securities. Follow these steps to
paper trade with Bitfinex:
1. Create a paper trading sub-account and refill the paper balance For instructions, see Paper Trading at Bitfinex on the
Bitfinex website.
2. Create an API key for your sub-account. For instructions, see How to create and revoke a Bitfinex API Key on the Bitfinex
website.
3. Use AccountType.Cash in your algorithms.
To paper trade securities other than TESTBTCTESTUSD and TESTBTCTESTUSDT, follow these steps to simulate paper trading
with the QuantConnect Paper Trading brokerage:
1. In the initialize method of your algorithm, set the Bitfinex brokerage model and your account type .
2. Deploy your algorithm with the QuantConnect Paper Trading brokerage .
Asset Classes
Our Bitfinex integration supports trading Crypto .
Data Providers
The QuantConnect data provider provides Crypto data during live trading.
Orders
We model the Bitfinex API by supporting several order types, order properties, and order updates. When you deploy live
algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our Bitfinex integration supports:
Order Type Crypto
Market
Limit
Stop market
Stop limit
Order Properties
We model custom order properties from the Bitfinex API. The following table describes the members of the
BitfinexOrderProperties object that you can set to customize order execution:
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
hidden bool
A flag to signal that the
order should be hidden.
Hidden orders do not
appear in the order book,
so they do not influence
other market participants.
Hidden orders incur the
taker fee.
post_only bool
A flag to signal that the
order must only add
liquidity to the order book
and not take liquidity from
the order book. If part of
the order results in taking
liquidity rather than
providing liquidity, the
order is rejected without
any part of it being filled.
Updates
We model the Bitfinex API by supporting order updates .
Fees
To view the Bitfinex trading fees, see the Fees Schedule page on the Bitfinex website. To view how we model their fees, see
Fees .
To use the Bitfinex brokerage in a live algorithm, the following table shows the fee settings you need on your Account > Fees
page on the Bitfinex website:
Fee Setting Value
Default currency for fees USD
Fee type for Exchange orders Currency Exchange Fee
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements.
Slippage
Orders through Bitfinex do not experience slippage in backtests. In Bitfinex paper trading and live trading, your orders may
experience slippage.
To view how we model Bitfinex slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests. In Bitfinex paper trading and live trading, if the quantity of your
market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is available in
the order book.
To view how we model Bitfinex order fills, see Fills .
Settlements
Trades settle immediately after the transaction
To view how we model settlement for Bitfinex trades, see Settlement .
Security and Stability
Note the following security and stability aspects of our Bitfinex integration.
Account Credentials
When you deploy live algorithms with Bitfinex, we don't save your brokerage account credentials.
API Outages
We call the Bitfinex API to place live trades. Sometimes the API may be down. Check the Bitfinex status page to see if the API is
currently working.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Bitfinex brokerage:
Virtual Pairs
All fiat and Crypto currencies are individual assets. When you buy a pair like BTCUSD, you trade USD for BTC. In this case, LEAN
removes some USD from your portfolio cashbook and adds some BTC. The virtual pair BTCUSD represents your position in that
trade, but the virtual pair doesn't actually exist. It simply represents an open trade. When you deploy a live algorithm, LEAN
populates your cashbook with the quantity of each currency, but it can't get your position of each virtual pair.
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
If you are deploying a paper trading algorithm without the QuantConnect Paper Trading brokerage, include the following
lines of code in the initialize method of your algorithm:
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Bitfinex Exchange from the drop-down menu.
4. Enter your API key and secret.
To generate your API credentials, see Account Types . Your account details are not saved on QuantConnect.
5. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
6. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
7. (Optional) Set up notifications .
8. Configure the Automatically restart algorithm setting.
 Charts  Statistics  Code Clone Algorithm
self.set_account_currency('TESTUSD') # or 'TESTUSDT'
self.set_brokerage_model(BrokerageName.BITFINEX, AccountType.CASH)
self.set_benchmark(lambda x: 0) # or the Symbol of the TESTBTCTESTUSD/TESTBTCTESTUSDT securities
PY
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
9. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Bloomberg EMSX
Brokerages
Bloomberg EMSX
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Terminal link icon
QuantConnect can integrate with the Bloomberg™ Server API (SAPI) or Desktop API (DAPI) in different cloud environments. This
integration allows research, backtesting, opitimization, and live trading through the Bloomberg APIs. Terminal Link is in no way
affiliated with or endorsed by Bloomberg™; it is simply an add-on. Add Terminal link to your organization to access the 1,300+
prime brokerages in the Bloomberg Execution Management System network.
QuantConnect Cloud only supports routing trades to the Bloomberg™ Server API. In this environment, you can route orders to
any of the prime brokerages that Bloomberg supports and you get to leverage the data, server management, and data
management from QuantConnect, giving you the best of both worlds. To use Terminal Link, you need to be a member of an
organization on the Trading Firm or Institution tier .
Account Types
Terminal Link supports order routing via the Bloomberg™ EMSX network. It's a margin account that's similiar to a FIX API, where
you set the buying power in the wizard when you're deploying to a professional prime brokerage account.
Data Providers
To use the Bloomberg™ EMSX network and SAPI on QuantConnect, you must use the QuantConnect data provider .
Asset Classes
Terminal Link supports trading the following asset classes:
Equities
Equity Options
Futures
Index Options
Orders
Terminal Link enables you to create and manage Bloomberg™ orders.
Order Types
The following table describes the available order types for each asset class that Terminal Link supports:
Order Type Equity Equity Options Futures Index Options
Market
Limit
Stop market
Stop limit
Order Properties
We model custom order properties from the Bloomberg EMSX API. The following table describes the members of the
TerminalLinkOrderProperties object that you can set to customize order execution:
Property Data Type Description Default Value
time_in_force TimeInForce
A TimeInForce instruction
to apply to the order. The
following instructions are
supported:
DAY
GOOD_TIL_CANCELED
good_til_date
TimeInForce.GOOD_TIL_CA
NCELED
notes str
The free form instructions
that may be sent to the
broker.
handling_instruction str
The instructions for
handling the order or route.
The values can be
preconfigured or a value
customized by the broker.
custom_notes_1 str
Custom user order notes 1.
For more information about
custom order notes, see
Custom Notes & Free Text
Fields in the EMSX API
documentation
custom_notes_2 str Custom user order notes 2.
custom_notes_3 str Custom user order notes 3.
custom_notes_4 str Custom user order notes 4.
custom_notes_5 str Custom user order notes 5.
account str The EMSX account.
broker str The EMSX broker code.
strategy StrategyParameters
An object that represents
the EMSX order strategy
details. You must append
strategy parameters in the
order that the EMSX API
expects. The following
strategy names are
supported: "DMA", "DESK",
"VWAP", "TWAP",
"FLOAT", "HIDDEN",
"VOLUMEINLINE",
"CUSTOM", "TAP",
"CUSTOM2",
"WORKSTRIKE",
"TAPNOW", "TIMED",
"LIMITTICK", "STRIKE"
execution_instruction str
The execution instruction
field.
automatic_position_side
s
bool
A flag that determines
whether to automatically
include the position side in
the order direction (buy-toopen, sell-to-close, etc.)
instead of the default (buy,
sell).
position_side OrderPosition/NoneType
An OrderPosition object
that specifies the position
side in the order direction
(buy-to-open, sell-to-close,
etc.) instead of the default
(buy, sell). This member
has precedence over
automatic_position_side
s
.
exchange Exchange
Defines the exchange name
for sending the order to.
For more information about the format that the Bloomberg EMSX API expects, see Create Order and Route Extended Request in
the EMSX API documentation and the createOrderAndRouteWithStrat documentation on the MathWorks website.
Get Open Orders
Terminal Link lets you access open orders .
Monitor Fills
Terminal Link allows you to monitor orders as they fill through order events .
Updates
Terminal Link doesn't support order updates , but you can cancel an existing order and then create a new order with the desired
arguments. For more information about this workaround, see the Workaround for Brokerages That Donʼt Support Updates .
Cancellations
Terminal Link enables you to cancel open orders .
Handling Splits
If you're using raw data normalization and you have active orders with a limit, stop, or trigger price in the market for a US Equity
when a stock split occurs, the following properties of your orders automatically adjust to reflect the stock split:
Quantity
Limit price
Stop price
Trigger price
Brokerage-Side Orders
By default, your algorithm doesn't record orders that you submit to your account by third-parties instead of through LEAN. To
accept these orders, create a custom brokerage message handler .
Fees
Orders filled with Terminal Link are subject to the fees of the Bloomberg™ Execution Management System and your prime
brokerage destination. To view how we model their fees, see Fees .
Margin
Set your cash and holdings state in the wizard when you deploy to the Bloomberg™ EMSX environment. We use these states to
model buying power and margin calls to ensure your algorithm stays within the margin requirements.
Fills
In live trading, LEAN routes your orders to the exchange or prime brokerage you select. The order fills and then routes back to
you.
To view how we model Bloomberg™ Execution Management System order fills, see Fills .
Compliance
Bloomberg™ is not affiliated with QuantConnect, nor does it endorse Terminal Link. A Bloomberg™ SAPI permission and EMSX
permission is required to use this brokerage connection, along with a Trading Firm or Institutional subscription on
QuantConnect.
The following rules apply:
All users of the integration must hold a Bloomberg License to be defined as an "Entitled User".
The Bloomberg SAPI will only be used for order routing and no data is permitted. The Bloomberg SAPI cannot be used for
black-box trading.
The following table shows the activities each of the Bloomberg technologies support:
Technology Research Backtesting Paper UAT Trading Live Trading
Server API
Set Up SAPI
The following few sections explain how to download the Bloomberg™ Server API (SAPI), install it on a cloud server, and add
firewall rules so it can connect to QuantConnect Cloud.
Download SAPI
Follow these steps to download the SAPI:
1. Install the Bloomberg™ Terminal .
2. Create a Bloomberg™ Terminal account .
3. In the Bloomberg™ Terminal, run WAPI<GO> .
4. On the API Developer's Help Site, click EMSX API .
5. On the EMSX API page, under the Server API Process section, click Link .
6. On the Server API Software Install page, click the correct download icons.
7. Click System Requirements .
Install the SAPI
Follow these steps to install the SAPI:
1. Spin up an E12x9 AWS instance or higher that your organization controls.
2. Run the SAPI installer on the cloud server.
For more information about this step, see How to install serverapi.exe in the EMSX API Programmers Guide. At the end of
the installion, you get a registration key.
3. Ask Bloomberg™ Support to activate your registration key.
4. Start the serverapi program.
On Windows, the default location is C: \ BLP \ ServerApi \ bin \ serverapi.exe .
Set Up Your Account
Follow these steps to set up your SAPI account:
1. Contact Bloomberg Support and ask them to enable the Server Side EMSX API.
2. Ask Bloomberg Support for your unique user identifier (UUID).
Save it somewhere safe. You will need it when you deploy live algorithms.
3. Contact the EMSX brokerage you plan to use and give them your UUID.
Add Firewall Rules
Follow these steps to configure the firewall rules on the AWS instance so that the SAPI can connect to QuantConnect Cloud:
1. Click Start .
2. Enter Windows Defender Firewall with Advanced Security and then press Enter .
3. In the left panel, click Inbound Rules .
4. In the right panel, click New Rule... .
5. Follow the prompts to create a program rule for the serverapi.
6. In the Windows Defender Firewall with Advanced Security window, double-click the serverapi row.
7. In the serverapi window, click the Scope tab.
8. In the Remote IP address section, add the QuantConnect Cloud IP address, 207.182.16.137.
9. Click OK .
10. Add the QuantConnect Cloud IP address to the other row in the table that has the serverapi name.
Deploy Live Algorithms
You need to set up the Bloomberg SAPI before you can deploy cloud algorithms with Terminal Link.
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Terminal Link from the drop-down menu.
4. Click the Connection Type field and then click SAPI from the drop-down menu.
5. In the Server Auth Id field, enter your unique user identifier (UUID).
The UUID is a unique integer identifier that's assigned to each Bloomberg Anywhere user. If you don't know your UUID,
contact Bloomberg.
6. In the EMSX Broker field, enter the EMSX broker to use.
7. In the Server Port field, enter the port where SAPI is listening.
The default port is 8194.
8. In the Server Host field, enter the public IP address of the SAPI AWS server.
9. In the EMSX Account field, enter the account to which LEAN should route orders.
10. In the EMSX Team field, enter the team account to receive events of your team's orders.
The default value is empty, which means LEAN disregards these notifications.
11. In the OpenFIGI Api Key field, enter your API key.
12. Click the Environment field and then click one of the options from the drop-down menu.
13. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
14. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
15. If your brokerage account has existing cash holdings, follow these steps ( see video ):
1. In the Algorithm Cash State section, click Show .
2. Click Add Currency .
3. Enter the currency ticker (for example, USD or CAD) and a quantity.
16. If your brokerage account has existing position holdings, follow these steps ( see video ):
1. In the Algorithm Holdings State section, click Show .
2. Click Add Holding .
3. Enter the symbol ID, symbol, quantity, and average price.
17. (Optional) Set up notifications .
18. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
19. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Trading Technologies
Brokerages
Trading Technologies
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Trading Technologies (TT) was founded by Gary Kemp in 1994 with the goal to create professional trading software,
infrastructure, and data solutions for a wide variety of users. TT provides access to trading Futures, Options, and Crypto. TT
also provides a charting platform, infrastructure services, and risk management tools. TT is not actually a brokerage. The firm is
a brokerage router with access to more than 30 execution destinations.
To view the implementation of the TT integration, see the Lean.Brokerages.TradingTechnologies repository .
Account Types
The TradingTechnologiesBrokerageModel does not have specific modeling for fees and slippage because TT is an order router
and can execute on many exchanges and brokerages. To set the brokerage model and account type in an algorithm, see the TT
brokerage model documentation . In live trading, TT reports the total fees of your orders after each order fill. Pass a different
BrokerageName to set_brokerage_model to backtest your algorithm with fee and slippage modeling. The brokerage model you
set should support the asset classes and orders in your algorithm.
Create an Account
Follow the account creation wizard on the TT website to create a TT account.
Paper Trading
Trading Technologies provides a separate User Acceptance Testing (UAT) Certification environment at uat.trade.tt . The TT
UAT environment connects to actual exchange certification environments for both market data and order routing.
To create a UAT account, follow the UAT account creation wizard on the TT website.
Create Deployment Credentials
After you create your TT account, see Create TT Users to create your live deployment credentials.
Asset Classes
Our TT integration supports trading Futures .
Data Providers
The QuantConnect data provider provides Futures data during live trading.
Orders
We model the TT API by supporting several order types, the TimeInForce order property, and order updates. When you deploy
live algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our TT integration supports:
Order Type Futures
Market
Limit
Stop market
Stop limit
TT enforces the following order rules:
If you are buying (selling) with a stop_market_order or a stop_limit_order , the stop price of the order must be greater
(less) than the current security price.
If you are buying (selling) with a stop_limit_order , the limit price of the order must be greater (less) than the stop price.
Time In Force
We model the TT API by supporting the DAY and GOOD_TIL_CANCELED TimeInForce order properties.
Updates
We model the TT API by supporting order updates .
Fees
To view the TT trading fees, see the Pricing page on the TT website. To view how we model their fees, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements.
Slippage
Orders through TT do not experience slippage in backtests and QuantConnect Paper Trading . In live trading, your orders may
experience slippage.
To view how we model TT slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model TT order fills, see Fills .
Security and Stability
Note the following security and stability aspects of our TT integration.
Account Credentials
When you deploy live algorithms with TT, we don't save your brokerage account credentials.
API Outages
We call the TT API to place live trades. Sometimes the API may be down. Check the TT status page to see if the API is currently
working.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the TT brokerage:
Create TT Users
Follow these steps to create your TT user name, account name, remote comp id, session password, app key, and app secret:
 Charts  Statistics  Code Clone Algorithm
Part 1: Sign in
On the TT website, sign in to your TT account or your TT UAT account . If you don't have an account, see Account Types to
create one.
Part 2: Add a New User
1. In the top navigation bar, click Setup .
2. On the Setup page, click +New User .
3. In the New User window, click the radio button that corresponds to user's employment status in your company and then
click Continue .
4. On the New User page, fill in the form with information about the new user.
The name that you put in the Username field is the username you will need when deploying live algorithms with LEAN.
In the Status section, click the Trade Mode field and then click TT Pro from the drop-down menu.
In the Advanced Settings section, select the Can create TT Rest API Key and Can create TT.NET SDK Client Side key check
boxes.
5. Click Create .
6. On the Setup page, click the new user in the table and then click Send Invitation .
Part 3: Create an App Key for the New User
1. On the Setup page, click the new user in the table and then click the App Keys tab.
2. In the App Keys section, click New .
3. In the Create New Application Key window, enter an application key name (for example, qb_alex_app_key ), select TT REST
API for application key type and then click Create .
4. Click Copy Secret to Clipboard and save it somewhere safe, like a text editor.
The text you copy contains the App Key and App Secret separated by a colon. For example, a79b17df-b249-45a1-9d12-
d5bbd2964626:424cc666-42aa-4125-a48e-13f630afd5a1.
5. Click OK .
Part 4: Create an Account for the New User
1. On the Setup page, in the left navigation bar, click Accounts .
2. On the Accounts page, click +New Account .
3. On the New User page, fill in the form with the following information:
Set the account name to something representative (for example, QC).
If there is a parent account, select it in the Parent field.
In the Parent field, select Routing (internal sub-account) .
4. Click Create .
5. On the Account page, click the new account in the table and then click the Users tab.
6. In the Users section, click +Add .
7. In the Select Users window, click the new user and then click Select .
8. Click Save Changes .
Part 5: Create a FIX Session for the New User
1. On the Setup page, in the left navigation bar, click FIX Sessions .
2. On the FIX Sessions page, click +New FIX Session .
3. On the New FIX Sessions page, fill in the form with the following information:
Set the FIX Session Name to something representative (for example, QC).
In the FIX Type field, select FIX Order Routing .
Set the Remote Comp ID to something representative (for example, qc_alex_id).
This Id is the remote comp id you will use when deploying live algorithms with LEAN.
The session password you set in this form is the password you will use when deploying live algorithms with LEAN.
Select the Send unsolicited order and fill message check box.
In the Status section, deselect the Inactive check box.
4. Click Create .
5. On the FIX Sessions page, click the new FIX session in the table and then click the Users tab.
6. In the Users section, click +Add .
7. In the Select Users window, click the new user and then click Select .
8. Click Save Changes .
Part 6: Verify the User Credentials
1. On the Setup page, in the left navigation bar, click Users .
2. On the Users page, click the new user in the table and then review the information under the FIX Sessions and App Keys
tabs.
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Trading Technologies from the drop-down menu.
4. Enter your TT user name, account name, remote comp id, session password, app key, and app secret.
To get your credentials, see Create TT Users .
Our TT integration routes orders via the TT FIX 4.4 Connection. Contact your TT representative to set the exchange where
you would like your orders sent. Your account details are not saved on QuantConnect.
Our integration fetches your positions using the REST endpoint, so the app key and app secret are your REST App
credentials.
5. Click the Environment field and then click one of the environments from the drop-down menu.
The following table shows the supported environments:
Environment Description
Live Trade in the production environment
UAT Trade in the User Acceptance Testing environment
6. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
7. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
8. If your brokerage account has existing cash holdings, follow these steps ( see video ):
1. In the Algorithm Cash State section, click Show .
2. Click Add Currency .
3. Enter the currency ticker (for example, USD or CAD) and a quantity.
9. (Optional) Set up notifications .
10. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
11. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Wolverine
Brokerages
Wolverine
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
Wolverine Execution Services is a diversified financial institution specializing in proprietary trading, asset management, order
execution services, and technology solutions. They are recognized as a market leader in derivatives valuation, trading, and
value-added order execution across global Equity, Options, and Futures markets. Their focus on innovation, achievement, and
integrity serves the interests of their clients and colleagues. Wolverine Execution Services is headquartered in Chicago, with
branch offices in New York, San Francisco, and London. They serve funds that have at least $5M assets under management.
To view the implementation of the Wolverine Execution Services brokerage integration, see the Lean.Brokerages.Wolverine
repository .
Account Types
Wolverine Execution Services supports cash and margin accounts. To set the account type in an algorithm, see the Wolverine
brokerage model documentation .
Create an Account
To create a Wolverine Execution Services account, contact their staff through the TradeWex website.
Paper Trading
Wolverine Execution Services doesn't support paper trading, but you can follow these steps to simulate it with QuantConnect:
1. In the initialize method of your algorithm, set the Wolverine brokerage model and your account type .
2. Deploy your algorithm with the QuantConnect Paper Trading brokerage .
Asset Classes
Our Wolverine Execution Services integration supports trading US Equities .
You may not be able to trade all assets with Wolverine. For example, if you live in the EU, you can't trade US ETFs. Check with
your local regulators to know which assets you are allowed to trade. You may need to adjust settings in your brokerage account
to live trade some assets.
Data Providers
The QuantConnect data provider provides US Equities data during live trading.
Orders
We model the Wolverine Execution Services API by supporting order types, but not order updates or extended market hours
trading. When you deploy live algorithms, you can place manual orders through the IDE.
Order Types
Our Wolverine Execution Services integration supports market orders .
Updates
We model the Wolverine Execution Services API by not supporting order updates.
Extended Market Hours
Wolverine Execution Services doesn't support extended market hours trading. If you place an order outside of regular trading
hours, the order is invalid.
Order Properties
We model custom order properties from the Wolverine API. The following table describes the members of the
WolverineOrderProperties object that you can set to customize order execution:
Property Description
time_in_force TimeInForce
exchange Exchange
exchange_post_fix str
Fees
Wolverine Execution Services charge $0.005 per share you trade. To view how we model their fees, see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements. If you have more than
$25,000 in your brokerage account, you can use the PatternDayTradingMarginModel to make use of the 4x intraday leverage
and 2x overnight leverage available on most brokerages from the PDT rule .
Slippage
Orders through Wolverine Execution Services do not experience slippage in backtests and QuantConnect Paper Trading . In live
trading, your orders may experience slippage.
To view how we model Wolverine slippage, see Slippage .
Fills
We fill market orders immediately and completely in backtests and QuantConnect Paper Trading . In live trading, if the quantity
of your market orders exceeds the quantity available at the top of the order book, your orders are filled according to what is
available in the order book.
To view how we model Wolverine Execution Services order fills, see Fills .
Settlements
If you trade with a margin account, trades settle immediately
To view how we model settlement for Wolverine trades, see Settlement .
Security and Stability
When you deploy live algorithms with Wolverine Execution Services, we don't save your brokerage account credentials.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the Wolverine Execution Services brokerage:
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
 Charts  Statistics  Code Clone Algorithm
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click Wolverine Execution Services from the drop-down menu.
4. Enter your Wolverine Execution Services credentials.
Your account details are not saved on QuantConnect.
5. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
6. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
7. If your brokerage account has existing cash holdings, follow these steps ( see video ):
1. In the Algorithm Cash State section, click Show .
2. Click Add Currency .
3. Enter the currency ticker (for example, USD or CAD) and a quantity.
8. If your brokerage account has existing position holdings, follow these steps ( see video ):
1. In the Algorithm Holdings State section, click Show .
2. Click Add Holding .
3. Enter the symbol ID, symbol, quantity, and average price.
9. (Optional) Set up notifications .
10. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
11. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > FIX Connections
Brokerages
FIX Connections
Introduction
The Financial Information eXchange (FIX) is the standard electronic communications protocol for front-office messaging. The
FIX community includes about 300 firms, including major investment banks.
Supported Connections
The following FIX connections are available on QuantConnect:
Name Integration Implementation Model Implementation
Raiffeisen Bank International Lean.Brokerages.RaiffeisenBankInternationRaBl IBrokerageModel.cs
Wolverine Lean.Brokerages.Wolverine WolverineBrokerageModel.cs
Live Trading > Brokerages > CFD and FOREX Brokerages
Brokerages
CFD and FOREX Brokerages
Introduction
QuantConnect enables you to run your algorithms in live mode with real-time market data.
QuantConnect integrates with OANDA for CFD and FOREX trading. OANDA was founded by Dr. Michael Stumm and Dr. Richard
Olsen in 1995 with the goal to "transform all aspects of how the world interacts with currencies, whether that be trading or
utilizing currency data and information". OANDA provides access to trading Forex and CFDs for clients in over 240 countries
and territories with no minimum deposit . OANDA also provides demo accounts, advanced charting tools, and educational
content
To view the implementation of the OANDA brokerage integration, see the Lean.Brokerages.OANDA repository .
Account Types
OANDA supports margin accounts. To set the account type in an algorithm, see the OANDA brokerage model documentation .
Create an Account
Follow the How to open an account page on the OANDA website to open an OANDA account.
You will need your account number and access token to deploy live algorithms. To get your account number, open the Account
Statement page on the OANDA website. Your account number is formatted as ###-###-######-### . To get your access
token, open the Manage API Access on the OANDA website.
Important note for European Union residents: On March 17th, 2023, OANDA Europe Markets Ltd. ("OEML") closed operations
and transferred accounts to OANDA TMS Brokers S.A. ("OANDA TMS")
1
. OANDA TWS does not offer REST API endpoints for
live trading. EU residents can trade with OANDA if they can open an account with another member of the OANDA Group, for
example, US citizens.
Paper Trading
OANDA supports paper trading. Follow these steps to set up an OANDA paper trading account:
1. Create an OANDA demo account .
2. Log in to your demo account.
3. On the Account page, in the My Services section, click Manage API Access .
4. On the Your key to OANDA's API page, click Generate .
Your access token displays. Store it somewhere safe. You need your access token to deploy an algorithm with your paper
trading account.
5. In the top navigation bar, click My Account .
6. On the Account page, in the Manage Funds section, click View .
7. On the My Funds page, in the Account Summary section, note your v20 Account Number.
You need your v20 Account Number to deploy an algorithm with your paper trading account.
Asset Classes
Our OANDA integration supports trading Forex and CFDs .
Data Providers
The QuantConnect data provider providers Forex and CFD trading data during live trading.
Orders
We model the OANDA API by supporting several order types, a TimeInForce order instruction, and order updates. When you
deploy live algorithms, you can place manual orders through the IDE.
Order Types
The following table describes the available order types for each asset class that our OANDA integration supports:
Order Type Forex CFD
Market
Limit
Stop market
Stop limit
Time In Force
We model the GOOD_TIL_CANCELED TimeInForce from the OANDA API.
Updates
We model the OANDA API by supporting order updates .
Fees
To view the OANDA trading fees, see the Our Charges and Fees page on the OANDA website. To view how we model their fees,
see Fees .
Margin
We model buying power and margin calls to ensure your algorithm stays within the margin requirements.
Slippage
Orders through OANDA do not experience slippage in backtests. In OANDA paper trading and live trading, your orders may
experience slippage.
To view how we model OANDA slippage, see Slippage .
Fills
To view how we model OANDA order fills, see Fills .
Settlements
Trades settle immediately after the transaction
To view how we model settlement for OANDA trades, see Settlement .
Security and Stability
Note the following security and stability aspects of our OANDA integration.
Account Credentials
When you deploy live algorithms with OANDA, we don't save your brokerage account credentials.
API Outages
We call the OANDA API to place live trades. Sometimes the API may be down. Check the OANDA status page to see if the API is
currently working.
Deposits and Withdrawals
You can deposit and withdraw cash from your brokerage account while you run an algorithm that's connected to the account.
We sync the algorithm's cash holdings with the cash holdings in your brokerage account every day at 7:45 AM Eastern Time
(ET).
Demo Algorithm
The following algorithm demonstrates the functionality of the OANDA brokerage:
Deploy Live Algorithms
You must have an available live trading node for each live trading algorithm you deploy.
Follow these steps to deploy a live algorithm:
1. Open the project you want to deploy.
2. Click the Deploy Live icon.
3. On the Deploy Live page, click the Brokerage field and then click OANDA from the drop-down menu.
4. Enter your OANDA account Id and access token.
 Charts  Statistics  Code Clone Algorithm
To get your account ID and access token, see the Create an Account section in the Account Types documentation. Your
account details are not saved on QuantConnect.
5. Click the Environment field and then click one of the environments.
The following table shows the supported environments:
Environment Description
Real Trade real money with fxTrade
Demo Trade paper money with fxTrade Practice
6. Click the Node field and then click the live trading node that you want to use from the drop-down menu.
7. (Optional) In the Data Provider section, click Show and change the data provider or add additional providers.
8. (Optional) Set up notifications .
9. Configure the Automatically restart algorithm setting.
By enabling automatic restarts , the algorithm will use best efforts to restart the algorithm if it fails due to a runtime error.
This can help improve the algorithm's resilience to temporary outages such as a brokerage API disconnection.
10. Click Deploy .
The deployment process can take up to 5 minutes. When the algorithm deploys, the live results page displays. If you know your
brokerage positions before you deployed, you can verify they have been loaded properly by checking your equity value in the
runtime statistics, your cashbook holdings, and your position holdings.
Live Trading > Brokerages > Unsupported Brokerages
Brokerages
Unsupported Brokerages
Introduction
New brokerages can be added if the brokerage has an API that is popular, stable, and officially supported by the brokerage. To
add a new brokerage to the platform, contact us .
Live Trading > Deployment
Live Trading
Deployment
Introduction
Deploy your trading algorithms live to receive real-time market data and submit orders on our co-located servers. As your
algorithms run, you can view their performance in the Algorithm Lab. Since the algorithms run in QuantConnect Cloud, you can
close the IDE without interrupting the execution of your algorithms. Deploying your algorithms to live trading through
QuantConnect is cheaper than purchasing server space, setting up data feeds, and maintaining the software on your own. To
deploy your algorithms on QuantConnect, you just need to follow the Deploy Live Algorithms section in the guide of your
brokerage .
Resources
Live trading nodes enable you to deploy live algorithms to our professionally-managed, co-located servers. You need a live
trading node for each algorithm that you deploy to our co-located servers. Several models of live trading nodes are available.
More powerful live trading nodes allow you to run algorithms with larger universes and give you more time for machine learning
training . Each security subscription requires about 5MB of RAM. The following table shows the specifications of the live trading
node models:
Name Number of Cores
Processing Speed
(GHz)
RAM (GB) GPU
L-MICRO 1 2.6 0.5 0
L1-1 1 2.6 1 0
L1-2 1 2.6 2 0
L2-4 2 2.6 4 0
L8-16-GPU 8 3.1 16 1/2
Refer to the Pricing page to see the price of each live trading node model.
To view the status of all of your organization's nodes, see the Resources panel of the IDE. When you deploy an algorithm, it uses
the best-performing resource by default, but you can select a specific resource to use .
The CPU nodes are available on a fair usage basis while the GPU nodes can be shared with a maximum of two members.
Depending on the server load, you may use all of the GPU's processing power. GPU nodes perform best on repetitive and
highly-parallel tasks like training machine learning models. It takes time to transfer the data to the GPU for computation, so if
your algorithm doesn't train machine learning models, the extra time it takes to transfer the data can make it appear that GPU
nodes run slower than CPU nodes.
Node Quotas
You need a live trading node for each simultaneous algorithm that you deploy. We do not support sub algorithms or sharing a
server with multiple algorithms. The tier of your organization determines the number of live trading nodes the organization can
have. The following number of live trading nodes are available for each tier:
Tier Node Quota
Free 0
Quant Researcher 2
Team 10
Trading Firm Unlimited
Institution Unlimited
To deploy multiple algorithms using a single brokerage, create sub-accounts in your brokerage account so that each algorithm
has its own set of brokerage connection credentials.
Ram Allocations
Members often use 8-32GB of RAM in backtesting and are concerned that their algorithms will not work in live trading since live
trading nodes have 512MB to 4GB of RAM. Backtesting nodes have more RAM because data is injected into your algorithm
roughly 100,000x faster during backtests than live trading. You use more RAM in backtesting because many data objects are
cached to achieve such fast speed. In live trading, 512MB to 4GB of RAM is sufficient for almost all use cases.
Wizard
Use the deployment wizard in the Algorithm Lab to deploy your algorithms to live trading . The deployment wizard lets you
select a brokerage, enter your brokerage credentials, select a data provider, select a live trading node, set up notifications, and
configure automatic algorithm restarts.
Most of the brokerages automatically load your cash holdings, position holdings, and submitted orders so that you can view
your portfolio state on the live results page . For brokerages that don't automatically load your holdings, you can enter your
cash and position holdings in the deployment wizard.
Unsupported Assets
If you have unsupported assets in your brokerage account when you deploy, Lean can't calculate the portfolio value correctly,
so margin calculations are wrong. To avoid issues, if your account has unsupported assets, Lean automatically exits on
deployment. For a list of supported assets, see the asset class dataset listing .
Automatic Restarts
Automatic restarts use best efforts to restart your algorithm if it fails due to a runtime error or an API disconnection. Automatic
restarts reduce the risk of your algorithm missing a trade during periods of downtime. If you enable automatic restarts when you
deploy your algorithm and your algorithm fails, your algorithm will try five times to restart. After five unsuccessful restarts, your
algorithm won't attempt to restart again. To prevent restarts due to coding bugs, algorithms only automatically restart if they
have been running for at least five minutes.
Security
Your code is stored in a database, isolated from the internet. When the code leaves the database, it is compiled and obfuscated
before being deployed to the cloud. If the cloud servers were compromised, this process makes it difficult to read your strategy.
As we've seen over recent years, there can never be any guarantee of security with online websites. However, we deploy all
modern and common security procedures. We deploy nightly software updates to keep the server up to date with the latest
security patches. We also use SSH key login to avoid reliance on passwords. Internally, we use processes to ensure only a
handful of people have access to the database and we always restrict logins to never use root credentials.
See our Security and IP documentation for more information.
Automate Deployments
If you have multiple deployments, use a notebook in the Research Enviroment to programmatically deploy, stop or liquidate
algorithms.
Best Practices
When you have a strategy that shows promising backtest results, consider paper trading the strategy before deploying it with
real money. Many of our brokerage integrations support a demo environment for paper trading. If your brokerage supports a
demo live environment, deploy a live algorithm that uses it. Otherwise, set the brokerage model to your brokerage and then
deploy your algorithm with the QuantConnect Paper Trading brokerage . The demo environment and reality model of your
brokerage provide the most accurate results for live trading.
While paper trading, perform the following stress tests to ensure your algorithm can handle interference:
Restart your algorithm when the market is open and closed.
Update and redeploy the algorithm .
Clone the project and deploy the cloned version.
If the preceding stress tests pass, load a small amount of money into your real money brokerage account for final validation.
When you're ready to transition to live trading, load the rest of your trading capital into the account that you already validated.
Live Trading > Notifications
Live Trading
Notifications
Introduction
Set up some live trading notifications so that you are notified of market events and your algorithm's performance. We support
email, SMS, webhooks, and Telegram notifications. If you set up notifications in the deployment wizard, we will notify you when
your algorithm places orders or emits insights. To be notified at other moments in your algorithm, create notifications in your
code files with the NotificationManager . Lean ignores notifications during backtests. To view the number of notification you
can send for free, see the Live Trading Notification Quotas .
Email
Email notifications can include up to 10KB of text content in the message body. These notifications can be slow since they go
through your email provider. If you don't receive an email notification that you're expecting, check your junk folders.
Follow these steps to set up email notifications in the deployment wizard:
1. On the Deploy Live page, enable at least one of the notification types.
The following table shows the supported notification types:
Notification Type Description
Order Events
Notifications for when the algorithm receives OrderEvent
objects
Insights
Notifications for when the algorithm emits Insight
objects
2. Click Email .
3. Enter an email address.
4. Enter a subject.
5. Click Add .
To add more email notifications, click Add Notification and then continue from step 2.
SMS
SMS notifications are the only type of notification that you don't need an internet connection to receive. They can include up to
1,600 characters of text content in the message body.
Follow these steps to set up SMS notifications in the deployment wizard:
1. On the Deploy Live page, enable at least one of the notification types.
The following table shows the supported notification types:
Notification Type Description
Order Events
Notifications for when the algorithm receives OrderEvent
objects
Insights
Notifications for when the algorithm emits Insight
objects
2. Click SMS .
3. Enter a phone number.
4. Click Add .
To add more SMS notifications, click Add Notification and then continue from step 2.
Telegram
Telegram notifications are automated messages to a Telegram group.
Follow these steps to set up Telegram notifications in the deployment wizard:
1. On the Deploy Live page, enable at least one of the notification types.
The following table shows the supported notification types:
Notification Type Description
Order Events
Notifications for when the algorithm receives OrderEvent
objects
Insights
Notifications for when the algorithm emits Insight
objects
2. Create a new Telegram group.
3. Add a bot to your Telegram group.
To create a bot, chat with @BotFather and follow its instructions. If you want to use our bot, the username is
@quantconnect_notifications_bot.
4. On the live deployment wizard, click Telegram .
5. Enter your user Id or group Id.
Your group Id is in the URL when you open your group chat in the Telegram web interface. For example, the group Id of
web.telegram.org/z/#-503016366 is -503016366.
6. If you are not using our notification bot, enter the token of your bot.
7. Click Add .
To add more Telegram notifications, click Add Notification and then continue from step 2.
Webhooks
Webhook notifications are an HTTP-POST request to a URL you provide. The request is sent with a timeout of 300s. You can
process these notifications on your web server however you want. For instance, you can inject the content of the notifications
into your server's database or use it to create other notifications on your own server.
Follow these steps to set up webhook notifications in the deployment wizard:
1. On the Deploy Live page, enable at least one of the notification types.
The following table shows the supported notification types:
Notification Type Description
Order Events
Notifications for when the algorithm receives OrderEvent
objects
Insights
Notifications for when the algorithm emits Insight
objects
2. Click Webhook .
3. Enter a URL.
4. If you want to add header information, click Add Header and then enter a key and value.
Repeat this step to add multiple header keys and values.
5. Click Add .
To add more webhook notifications, click Add Notification and then continue from step 2.
Quotas
The number of email, Telegram, or webhook notifications you can send in each live algorithm for free depends on the tier of
your organization. The following table shows the hourly quotas:
Tier Number of Notifications Per Hour
Free N/A
Quant Researcher 20
Team 60
Trading Firm 240
Institution 3,600
If you exceed the hourly quota, each additional email, Telegram, or webhook notification costs 1 QuantConnect Credit (QCC).
Each SMS notification you send to a US or Canadian phone number costs 1 QCC. Each SMS notification you send to an
international phone number costs 10 QCC.
Terms of Use
The notification system can't be used for data distribution.
Live Trading > Results
Live Trading
Results
Introduction
The live results page shows your algorithm's live trading performance. Review the results page to see how your algorithm has
been performing and to investigate ways to improve it.
View Live Results
The live results page automatically displays when you deploy a live algorithm . The page presents the algorithm's equity curve,
holdings, trades, logs, server statistics, and much more information.
The content in the live results page updates as your algorithm executes. You can close or refresh the window without
interrupting the algorithm because the live trading node processes on our servers. If you close the page, you can view all of
your live projects to open the page again.
Runtime Statistics
The banner at the top of the live results page displays the performance statistics of your algorithm.
The following table describes the default runtime statistics:
Statistic Description
Equity The total portfolio value if all of the holdings were sold at current market rates.
Fees The total quantity of fees paid for all the transactions.
Holdings The absolute sum of the items in the portfolio.
Net Profit The dollar-value return across the entire trading period.
PSR
The probability that the estimated Sharpe ratio of an algorithm is greater than a benchmark
(1).
Return The rate of return across the entire trading period.
Unrealized
The amount of profit a portfolio would capture if it liquidated all open positions and paid the
fees for transacting and crossing the spread.
Volume The total value of assets traded for all of an algorithm's transactions.
To add a custom runtime statistic, see Add Statistics .
If you stop and redeploy a live algorithm, the runtime statistics are reset.
Built-in Charts
The live results page displays the equity curve of your algorithm so that you can analyze its performance in real-time.
The following table describes the series in the Strategy Equity chart:
Series Description
Equity The live equity curve of your algorithm.
Out of Sample Backtest The backtest equity curve of your algorithm during the live trading period.
Meta
Points in time when you deployed your algorithm, stopped your algorithm, and when your
algorithm encountered a runtime error.
The following table describes the other charts displayed on the page:
if ($cloudPlatform) { ?> } ?>
Chart Description
Drawdown A time series of equity peak-to-trough value.
Exposure A time series of long and short exposure ratios.
Assets Sales Volume A chart showing the proportion of total volume for each traded security.
Portfolio Margin
A stacked area chart of the portfolio margin usage. For more information about this chart,
see Portfolio Margin Plots .
Asset Plot A time series of an asset's price with order event annotations. For more information about
these charts, see Asset Plots .
Asset Plots
Asset plots display the trade prices of an asset and the following order events you have for the asset:
Order Event Icon
Submissions Gray circle
Updates Blue circle
Cancellations Gray square
Fills and partial fills Green (buys) or red (sells) arrows
The following image shows an example asset plot for AAPL:
The order submission icons aren't visible by default.
View Plots
Follow these steps to open an asset plot:
1. Open the live results page.
2. Click the Orders tab.
3. Click the Asset Plot icon that's next to the asset Symbol in the Orders table.
Tool Tips
When you hover over one of the order events in the table, the asset plot highlights the order event, displays the asset price at
the time of the event, and displays the tag associated with the event. Consider adding helpful tags to each order event to help
with debugging your algorithm. For example, when you cancel an order, you can add a tag that explains the reason for
cancelling it.
Adjust the Display Period
The resolution of the asset price time series in the plot doesn't necessarily match the resolution you set when you subscribed to
the asset in your algorithm. If you are displaying the entire price series, the series usually displays the daily closing price.
However, when you zoom in, the chart will adjust its display period and may use higher resolution data. To zoom in and out,
perform either of the following actions:
Click the 1m , 3m , 1y , or All period in the top-right corner of the chart.
Click a point on the chart and drag your mouse horizontally to highlight a specific period of time in the chart.
If you have multiple order events in a single day and you zoom out on the chart so that it displays the daily closing prices, the
plot aggregates the order event icons together as the price on that day.
Order Fill Prices
The plot displays fill order events at the actual fill price of your orders. The fill price is usually not equal to the asset price that
displays because of the following reasons:
Your order experiences slippage .
If you use quote data, your order fills at the bid or ask price.
The fill model may fill your order at the high or low price.
Custom Charts
The results page shows the custom charts that you create.
Supported Chart Types
We support the following types of charts:
If you use SeriesType.Candle and plot enough values, the plot displays candlesticks. However, the plot method only accepts
one numerical value per time step, so you can't plot candles that represent the open, high, low, and close values of each bar in
your algorithm. The charting software automatically groups the data points you provide to create the candlesticks, so you can't
control the period of time that each candlestick represents.
To create other types of charts, save the plot data in the Object Store and then load it into the Research Environment. In the
Research Environment, you can create other types of charts with third-party charting packages .
Supported Markers
When you create scatter plots, you can set a marker symbol. We support the following marker symbols:
Chart Sampling
Charts are sampled every one and ten minutes. If you create 1-minute resolution custom charts, the IDE charting will downgrade
the granularity and display the 10-minutes sampling after a certain amount of samples.
Demonstration
For more information about creating custom charts, see Charting .
Adjust Charts
You can manipulate the charts displayed on the live results page.
Toggle Charts
To display and hide a chart on the live results page, in the Select Chart section, click the name of a chart.
Toggle Chart Series
To display and hide a series on a chart on the live results page, click the name of a series at the top of a chart.
Adjust the Display Period
To zoom in and out of a time series chart on the live results page, perform either of the following actions:
Click the 1m , 3m , 1y , or All period in the top-right corner of the chart.
Click a point on the chart and drag your mouse horizontally to highlight a specific period of time in the chart.
If you adjust the zoom on a chart, it affects all of the charts.
After you zoom in on a chart, slide the horizontal bar at the bottom of the chart to adjust the time frame that displays.
Resize Charts
To resize a chart on the live results page, hover over the bottom-right corner of the chart. When the resize cursor appears, hold
the left mouse button and then drag to the desired size.
Move Charts
To move a chart on the live results page, click, hold, and drag the chart title.
Refresh Charts
Refreshing the charts on the live results page resets the zoom level on all the charts. If you refresh the charts while your
algorithm is executing, only the data that was seen by the Lean engine after you refreshed the charts is displayed. To refresh
the charts, in the Select Chart section, click the reset icon.
Holdings
The Holdings tab on the live results page displays your positions and cash.
The following table describes the properties that display for each of your positions:
Property Description
Symbol The ticker of the security.
Average Price The average price that you paid for the position.
Quantity The size of your position.
Market Value The value of your position if sold with market orders.
Unrealized The unrealized profit of your position, including fees and spread costs.
The values in the positions section update as new data points are injected into your algorithm. The cash section displays the
quantity of each currency in your algorithm's CashBook . View the Holdings tab to see your holdings, add security subscriptions ,
and place manual orders . To view all of your current holdings and active data subscriptions, enable the Show All Portfolio check
box.
Orders
The live results page displays the orders of your algorithm and you can download them to your local machine.
View in the GUI
To see the orders that your algorithm created, open the live results page and then click the Orders tab. If there are more than 10
orders, use the pagination tools at the bottom of the Orders Summary table to see all of the orders. Click on an individual order
in the Orders Summary table to reveal all of the order events , which include:
Submissions
Fills
Partial fills
Updates
Cancellations
Option contract exercises and expiration
The timestamps in the Order Summary table are based in Eastern Time (ET).
Access the Order Summary CSV
To view the orders data in CSV format, open the live results page, click the Orders tab, and then click Download Orders . The
content of the CSV file is the content displayed in the Orders Summary table when the table rows are collapsed. The timestamps
in the CSV file are based in Coordinated Universal Time (UTC).
Access in Jupyter Notebooks
To programmatically analyze orders, call the read_live_orders method or the /live/orders/read endpoint.
Insights
The live results page displays the insights of your algorithm and you can download them to your local machine.
View in the GUI
To see the insights your algorithm emit, open the live result page and then click the Insights tab. If there are more than 10
insights, use the pagination tools at the bottom of the Insights Summary table to see all of the insights. The timestamps in the
Insights Summary table are based in Eastern Time (ET).
Download JSON
To view the insights in JSON format, open the live result page, click the Insights tab, and then click Download Insights . The
timestamps in the CSV file are based in Coordinated Universal Time (UTC).
Logs
The Logs tab on the live results page displays all of the logging statements and status messages your algorithm creates. Their
timestamps in the log file are in Coordinated Universal Time (UTC). The status messages include all of the points in time when
your algorithm deployed, encountered an error, sent an order, or quit executing. It's good practice to add logs in live algorithms
because then you can see what is happening while it executes. If you stop and redeploy your algorithm, the logs are retained.
You can view the log file on the live results page or download them to your local machine.
View in the GUI
To see the log file your algorithm has created, open the live results page and then click the Logs tab.
To filter the logs, enter a search string in the Filter logs field.
Download Log File
To download the log file, open the live result page, click the Logs tab, and then click Download Logs .
Project Files
The live results page displays the project files used to deploy the algorithm. To view the files, click the Code tab. By default, the
main.py file displays. To view other files in the project, click the file name and then select a different file from the drop-down
menu.
To create a new project with the project files used to deploy the algorithm, click Clone Algorithm .
View All Live Projects
The Your Strategies section of the Strategy Explorer page displays the status of all the live algorithms in your organizations. To
view the page, log in to the Algorithm Lab and then, in the left navigation bar, click Strategy Explorer .
Errors
If your live algorithm throws a runtime error, it stops executing and we send you an email. If you enabled automatic restarts
when you deployed your algorithm, your algorithm will try five times to restart.
Share Results
You can share your live results with anyone, even if they don't have a QuantConnect account. To share your results, follow
these steps:
1. In the Share Results section of the live results page, click Make Public .
2. To get a URL that you can share with others, click Live Stream .
3. To get an iframe that you can embed on a website, click Embed Code .
The theme parameter of the URL defines the color theme. Valid values are darkly (dark) or chrome (light).
To stop sharing your live results, in the Share Results section of the live results page, click Make Private .
Live Trading > Algorithm Control
Live Trading
Algorithm Control
Introduction
The algorithm control features on the live results page let you adjust your algorithm while it is executing live so that you can
perform actions that are not written in the project files. The control features let you intervene in the execution of your algorithm
and make adjustments. For instance, you can create security subscriptions, place trades, stop the algorithm, and update the
algorithm.
Add Security Subscriptions
The live results page enables you to manually create security subscriptions for your algorithm instead of calling the
Add securityType methods in your code files. If you add security subscriptions to your algorithm, you can place manual trades
through the IDE without having to edit and redeploy the algorithm. Follow these steps to add security subscriptions:
1. Open your algorithm's live results page .
2. In the Holdings tab, click Add Security .
3. Enter the symbol, security type, resolution, leverage, and market of the security you want to add.
4. If you want the data for the security to be filled-forward, check the Fill Forward check box.
5. If you want to subscribe to extended market hours for the security, check the Extended Market Hours check box.
6. Click Add Security .
You can't manually remove security subscriptions from the IDE.
Place Manual Trades
The live results page lets you manually place orders instead of calling the automated methods in your project files. You can use
any order type that is supported by the brokerage that you used when deploying the algorithm. To view the supported order
types of your brokerage, see the Orders section of your brokerage model . Some example situations where it may be helpful to
place manual orders instead of stopping and redeploying the algorithm include the following:
Your brokerage account had holdings in it before you deployed your algorithm
Your algorithm had bugs in it that caused it to purchase the wrong security
You want to add a hedge to your portfolio without adjusting the algorithm code
You want to rebalance your portfolio before the rebalance date
Note that it's not currently possible to cancel manual orders.
Follow these steps to place manual orders:
1. Open your algorithm's live results page .
2. In the Holdings tab, if the security you want to trade isn't listed, click Show All Portfolio .
3. If the security you want to trade still isn't listed, subscribe to the security .
4. Click the security you want to trade.
5. Click Create Order or Liquidate .
6. If you clicked Create Order , enter an order quantity.
7. Click the Type field and then click an order type from the drop-down menu.
8. Click Submit Order .
Liquidate Positions
The live results page has a Liquidate button that acts as a "kill switch" to sell all of your portfolio holdings. If your algorithm has a
bug in it that caused it to purchase a lot of securities that you didn't want, this button let's you easily liquidate your portfolio
instead of placing many manual trades. When you click the Liquidate button, if the market is open for an asset you hold, the
algorithm liquidates it with market orders. If the market is not open, the algorithm places market on open orders. After the
algorithm submits the liquidation orders, it stops executing.
Follow these steps to liquidate your positions:
1. Open your algorithm's live results page .
2. Click Liquidate .
3. Click Liquidate again.
Stop the Algorithm
The live trading results page has a Stop button to immediately stop your algorithm from executing. When you stop a live
algorithm, your portfolio holdings are retained. Stop your algorithm if you want to perform any of the following actions:
Update your project's code files
Upgrade the live trading node
Update the settings you entered into the deployment wizard
Place manual orders through your brokerage account instead of the web IDE
Furthermore, if you receive new securities in your portfolio because of a reverse merger, you also need to stop and redeploy the
algorithm.
LEAN actively terminates live algorithms when it detects interference outside of the algorithm's control to avoid conflicting race
conditions between the owner of the account and the algorithm, so avoid manipulating your brokerage account and placing
manual orders on your brokerage account while your algorithm is running. If you need to adjust your brokerage account
holdings, stop the algorithm, manually place your trades, and then redeploy the algorithm.
Follow these steps to stop your algorithm:
1. Open your algorithm's live results page .
2. Click Stop .
3. Click Stop again.
Update the Algorithm
If you need to adjust your algorithm's project files or parameter values , stop your algorithm, make your changes, and then
redeploy your algorithm. You can't adjust your algorithm's code or parameter values while your algorithm executes.
When you stop and redeploy a live algorithm, your project's live results is retained between the deployments. To clear the live
results history, clone the project and then redeploy the cloned version of the project.
To update parameters in live mode, add a Schedule Event that downloads a remote file and uses its contents to update the
parameter values.
def initialize(self):
self.parameters = { }
if self.live_mode:
def download_parameters():
content = self.download(url_to_remote_file)
# Convert content to self.parameters
self.schedule.on(self.date_rules.every_day(), self.time_rules.every(timedelta(minutes=1)),
download_parameters)
PY
Live Trading > Reconciliation
Live Trading
Reconciliation
Introduction
Algorithms usually perform differently between backtesting and live trading over the same time period. Backtests are
simulations where we model reality as close as possible, but the modeling isn't always perfect. To measure the performance
differences, we run an out-of-sample (OSS) backtest in parallel to all of your live trading deployments. The live results page
displays the live equity curve and the OOS backtest equity curve of your algorithms.
If your algorithm is perfectly reconciled, it has an exact overlap between its live and OOS backtest equity curves. Deviations
mean that the performance of your algorithm has differed between the two execution modes. Several factors can contribute to
the deviations.
Differences From Data
The data that your algorithm uses can cause differences between backtesting and live trading performance.
Look-Ahead Bias
The Time Frontier minimizes the risk of look-ahead bias in backtests, but it does not completely eliminate the risk of look-ahead
bias. For instance, if you use a custom dataset that contains look-ahead bias, your algorithm's live and backtest equity curves
may deviate. To avoid look-ahead bias with custom datasets, set a period on your custom data points so that your algorithm
receives the data points after the time + period .
Discrete Time Steps
In backtests, we inject data into your algorithm at predictable times, according to the data resolution. In live trading, we inject
data into your algorithm when new data is available. Therefore, if your algorithm has a condition with a specific time (i.e. time is
9:30:15), the condition may work in backtests but it will always fail in live trading since live data has microsecond precision. To
avoid issues, either use a time range in your condition (i.e. 9:30:10 < time < 9:30:20), use a rounded time, or use a Scheduled
Event.
Custom Data Emission Times
Custom data is often timestamped to midnight, but the data point may not be available in reality until several days after that
point. If your custom dataset is prone to this delay, your backtest may not fetch the same data at the same time or frequency
that your live trading algorithm receives the data, leading to deviations between backtesting and live trading. To avoid issues,
ensure the timestamps of your custom dataset are the times when the data points would be available in reality.
In backtesting, LEAN and custom data are perfectly synchonized. In live trading, daily and hourly data from a custom data
source are not because of the frequency that LEAN checks the data source depends on the resolution argument. The
following table shows the polling frequency of each resolution:
Resolution Update Frequency
Daily Every 30 minutes
Hour Every 30 minutes
Minute Every minute
Second Every second
Tick Constantly checks for new data
Split Adjustment of Indicators
Backtests use adjusted price data by default. Therefore, if you don't change the data normalization mode , the indicators in your
backtests are updated with adjusted price data. In contrast, if a split or dividend occurs in live trading, your indicators will
temporarily contain price data from before the corporate event and price data from after the corporate event. If this occurs, your
indicators will produce different signals in your backtests compared to your live trading deployment. To avoid issues, reset and
warm up your indicators when your algorithm receives a corporate event.
Tick Slice Sizes
In backtesting, we collect ticks into slices that span 1 millisecond before injecting them into your algorithm. In live trading, we
collect ticks into slices that span up to 70 milliseconds before injecting them into your algorithm. This difference in slice sizes
can cause deviations between your algorithm's live and OOS backtest equity curves. To avoid issues, ensure your strategy logic
is compatible with both slice sizes.
Differences From Modeling
The modeling that your algorithm uses can cause differences between backtesting and live trading performance.
Reality Modeling Error
We provide brokerage models to model fees, slippage, and order fills in backtests. However, these model predictions may not
always match the fees that your live algorithm incurs, leading to deviations between backtesting and live trading. You can adjust
the reality models that your algorithm uses to more accurately reflect the specific assets that you're trading. For more
information about reality models, see Reality Modeling .
Market Impact
We don't currently model market impact. So, if you are trading large orders, your fill prices can be better during backtesting than
live trading, causing deviations between backtesting and live trading. To avoid issues, implement a custom fill model in your
backtests that incorporates market impact.
Fills
In backtests, orders fill immediately. In live trading, they are sent to your brokerage and take about half a second to execute. If
you fill an order in a backtest with stale data, deviations between backtesting and live trading can occur because the order is
filled at a price that is likely different from the real market price. Stale order fills commonly occur in backtests when you create a
Scheduled Event with an incompatible data resolution. For instance, if you subscribe to hourly data, place a Scheduled Event for
11:15 AM, and fill an order during the Scheduled Event, the order will fill at a stale price because the data between 11:00 AM and
11:15 AM is missing. To avoid stale fills, only place orders when your algorithm receives price data.
In live trading, your brokerage provides the fill price of your orders. Since the backtesting brokerage models do not know the
price at which live orders are filled, the fill price of backtest orders is based on the best price available in the current backtesting
data. Similarly, limit orders can fill at different prices between backtesting and live trading. In backtesting, limit orders fill as soon
as the limit price is hit. In live trading, your brokerage may fill the same limit order at a different price or fail to fill the order,
depending on the position of your order in their order book.
Borrowing Costs
We do not currently simulate the cost of borrowing shorts in backtests. Therefore, if your algorithm takes short positions,
deviations can occur between backtesting and live trading. We are working on adding the functionality to model borrowing fees.
Subscribe to GitHub Issue #4563 to track the feature progress.
Differences From Brokerage
The brokerage that your algorithm uses can cause differences between backtesting and live trading performance.
Portfolio Allocations on Small Accounts
If you trade a small portfolio, it's difficult to achieve accurate portfolio allocations because shares are usually sold in whole
numbers. For instance, you likely can't allocate exactly 10% of your portfolio to a security. You can use fractional shares to
achieve accurate portfolio allocations, but not all brokerages support fractional shares. To get the closest results when
backtesting and live trading over the same period, ensure both algorithms have the same starting cash balance.
Different Backtest Parameters
If you don't start your backtest and live deployment on the same date with the same holdings, deviations can occur between
backtesting and live trading. To avoid issues, ensure your backtest parameters are the same as your live deployment.
Non-deterministic State From Algorithm Restarts
If you stop and redeploy your live trading algorithm, it needs to restart in a stateful way or else deviations can occur between
backtesting and live trading. To avoid issues, redeploy your algorithm in a stateful way using the set_warm_up and history
methods. Furthermore, use the Object Store to save state information between your live trading deployments.
Existing Portfolio Securities
If you deploy your algorithm to live trading with a brokerage account that has existing holdings, your live trading equity curve
reflects your existing positions, but the backtesting curve won't. Therefore, if you have existing positions in your brokerage
account when you deploy your algorithm to live trading, deviations will occur between backtesting and live trading. To avoid
issues, deploy your algorithm to live trading using a separate brokerage account or subaccount that does not have existing
positions.
Brokerage Limitations
We provide brokerage models that support specific order types and model your buying power. In backtesting, we simulate your
orders with the brokerage model you select. In live trading, we send your orders to your brokerage for execution. If the
brokerage model that you use in backtesting is not the same brokerage that you use in live trading, deviations may occur
between backtesting and live trading. The deviations can occur if your live brokerage doesn't support the order types that you
use or if the backtesting brokerage models your buying power with a different methodology than the real brokerage. To avoid
brokerage model issues, set the brokerage model in your backtest to the same brokerage that you use in live trading.
Live Trading > Risks
Live Trading
Risks
Introduction
There are risks associated with deploying your algorithms to live trading. Strategy, portfolio, market, counterparty, operational,
and error risks can cause you to lose capital. Some of these risks can be out of your control, but there are ways that you can
mitigate them.
Strategy
Strategy risk is the risk that results from designing a strategy based on a statistical model. If you ignore the underlying
assumptions of the statistical model, you are exposed to strategy risk. Even if you test that the model assumptions are held, if
the market environment changes, the new environment may violate the underlying assumptions of the model after you have
deployed it to live trading. Additionally, your strategy development process may be prone to overfitting, survivorship bias, or
look-ahead bias , which increases your exposure to strategy risk. To address strategy risk, use rolling parameters when training
your statistical models and perform the required statistical tests before training such models.
Portfolio
Portfolio risk is the risk associated with your portfolio as a whole. For instance, you're exposed to portfolio risk if you allocate
too much of your portfolio to a particular factor, the capacity of your trading strategies reduces, or the correlation of the
strategies in your portfolio increases. To address portfolio risk, diversify your portfolio among multiple factors, monitor the
rolling capacity of your trading strategies, and frequently check the correlation of your trading strategies.
Market
Market risk, also known as systematic risk, is the risk that the value of your portfolio will decrease due to the value of the entire
market decreasing. Market risk is caused by changes in interest rates, changes in currency exchange rates, geopolitical events,
natural disasters, wars, terrorist attacks, and economic recessions. Additionally, central bank announcements and changes to
monetary policy can increase overall market volatility and market risk. To address market risk, you can increase diversification,
reduce your portfolio beta , hedge your positions with put Options, or hedge against volatility with volatility index securities.
Counterparty
Counterparty risk is the risk that a counterparty with which you engage won't pay an obligation that they have made with you.
Most commonly, counterparty risk is associated with the risk that your brokerage goes out of business without returning the
trading capital that you have in your brokerage account. Brokerages can go bankrupt just like any other business. To address
counterparty risk, diversify your portfolio across multiple brokers that have a strong reputation. If you allocate your capital
across multiple brokers and one of them goes out of business, you won't lose all of your trading capital.
Operational
Operational risks are the risks within your fund that relate to business operations, such as business risks, regulatory risks,
trading infrastructure risks, and the risks of employees committing fraud or quitting. Operational risks are a result of the nature
of a trading business, having employees, and regulatory changes. To address operational risks, stay up to date on potential
regulatory changes, only hire employees that have signed contracts that protect your firm, use open-source trading
infrastructure that's maintained by experts (Lean), and use co-located servers so that you don't need to tend to hardware
failures and internet outages.
Error
Error risk is the risk associated with errors occurring in your strategy logic or trading infrastructure. Error risks occur because
bugs naturally arise in trading algorithms and the underlying engine that the algorithms use to execute. The Lean trading engine
has been under constant development for over 10 years, but there are always more improvements that can be implemented. To
address error risk, backtest your trading algorithm before deploying it live to test if it has coding errors, stay up to date on the
Lean GitHub Issues , and have close access to your email at all times. If your trading algorithm fails, we notify you through email.
You can also enable automatic restarts when you deploy algorithms.
Optimization
Optimization
Parameter optimization is the process of finding the optimal algorithm parameters to maximize or minimize an objective
function. For instance, you can optimize your indicator parameters to maximize the Sharpe ratio that your algorithm achieves
over a backtest. Optimization can help you adjust your strategy to achieve better backtesting performance, but be wary of
overfitting. If you select parameter values that model the past too closely, your algorithm may not be robust enough to perform
well using out-of-sample data.
Getting Started
Learn the basics
Parameters
Variables being optimized
Objectives
Target metric of performance
Strategies
How parameters are adjusted
Deployment
Run optimization jobs
Results
Your performance dashboard
See Also
Running Optimizations
Reviewing Results
Optimization > Getting Started
Optimization
Getting Started
Introduction
Parameter optimization is the process of finding the optimal algorithm parameters to maximize or minimize an objective
function. For instance, you can optimize your indicator parameters to maximize the Sharpe ratio that your algorithm achieves
over a backtest. Optimization can help you adjust your strategy to achieve better backtesting performance, but be wary of
overfitting. If you select parameter values that model the past too closely, your algorithm may not be robust enough to perform
well using out-of-sample data.
Launch Optimization Jobs
The following video demonstrates how to launch an optimization job:
You need the following to optimize parameters:
At least one algorithm parameter in your project .
The GetParameter method in your project.
A successful backtest of the project.
QuantConnect Credit (QCC) in your organization.
Follow these steps to optimize parameters:
1. Open the project that contains the parameters you want to optimize.
2. In the top-right corner of the IDE, click the Optimize icon.
3. On the Optimization page, in the Parameter & Constraints section, enter the name of the parameter to optimize.
The parameter name must match a parameter name in the Project panel.
4. Enter the minimum and maximum parameter values.
5. Click the gear icon next to the parameter and then enter a step size.
6. If you want to add another parameter to optimize, click Add Parameter .
You can optimize a maximum of three parameters. To optimize more parameters, run local optimizations with the CLI .
7. If you want to add optimization constraints , follow these steps:
1. Click Add Constraint .
2. Click the target field and then select a target from the drop-down menu.
3. Click the operation field and then an operation from the drop-down menu.
4. Enter a constraint value.
8. In the Estimated Number and Cost of Backtests section, click an optimization node and then select a maximum number of
nodes to use.
9. In the Strategy & Target section, click the Choose Optimization Strategy field and then select a strategy from the dropdown menu.
10. Click the Select Target field and then select a target from the drop-down menu.
The target (also known as objective) is the performance metric the optimizer uses to compare the backtest performance of
different parameter values.
11. Click Maximize or Minimize to maximize or minimize the optimization target, respectively.
12. Click Launch Optimization .
The optimization results page displays. As the optimization job runs, you can close or refresh the window without
interrupting the job because the nodes are processing on our servers.
To abort a running optimization job, in the Status panel, click Abort and then click Yes .
View Individual Backtest Results
The optimization results page displays a Backtests table that includes all of the backtests that ran during the optimization job.
The table lists the parameter values of the backtests in the optimization job and their resulting values for the objectives.
Open the Backtest Results Page
To open the backtest result page of one of the backtests in the optimization job, click a backtest in the table.
Download the Table
To download the table, right-click one of the rows, and then click Export > CSV Export .
Filter the Table
Follow these steps to apply filters to the Backtests table:
1. On the right edge of the Backtests table, click Filters .
2. Click the name of the column to which you want the filter to be applied.
3. If the column you selected is numerical, click the operation field and then select one of the operations from the drop-down
menu.
4. Fill the fields below the operation you selected.
Toggle Table Columns
Follow these steps to hide and show columns in the Backtests table:
1. On the right edge of the Backtests table, click Columns .
2. Select the columns you want to include in the Backtests table and deselect the columns you want to exclude.
Sort the Table Columns
In the Backtests table, click one of the column names to sort the table by that column.
View All Optimizations
Follow these steps to view all of the optimization results of a project:
1. Open the project that contains the optimization results you want to view.
2. At the top of the IDE, click the Results icon.
A table containing all of the backtest and optimization results for the project is displayed. If there is a play icon to the left of
the name, it's a backtest result . If there is a fast-forward icon next to the name, it's an optimization result .
3. (Optional) In the top-right corner, select the Show field and then select one of the options from the drop-down menu to
filter the table by backtest or optimization results.
4. (Optional) In the bottom-right corner, click the Hide Error check box to remove backtest and optimization results from the
table that had a runtime error.
5. (Optional) Use the pagination tools at the bottom to change the page.
6. (Optional) Click a column name to sort the table by that column.
7. Click a row in the table to open the results page of that backtest or optimization.
Rename Optimizations
We give an arbitrary name (for example, "Smooth Apricot Chicken") to your optimization result files, but you can follow these
steps to rename them:
1. Hover over the optimization you want to rename and then click the pencil icon that appears.
2. Enter the new name and then press Enter .
Delete Optimizations
Hover over the optimization you want to delete and then click the trash can icon that appears to delete the optimization result.
On-Premise Optimizations
For information about on-premise optimizations with Local Platform , see Getting Started .
Get Optimization Id
To get the optimization Id, open the optimization result page and then scroll down to the table that shows the individual backtest
results . The optimization Id is at the top of the table. An example optimization Id is O-696d861d6dbbed45a8442659bd24e59f.
Optimization > Parameters
Optimization
Parameters
Introduction
Parameters are project variables that your algorithm uses to define the value of internal variables like indicator arguments or the
length of lookback windows.
Parameters are stored outside of your algorithm code, but we inject the values of the parameters into your algorithm when you
launch an optimization job . The optimizer adjusts the value of your project parameters across a range and step size that you
define to minimize or maximize an objective function. To optimize some parameters, add some parameters to your project and
add the get_parameter method to your code files.
Set Parameters
Algorithm parameters are hard-coded values for variables in your project that are set outside of the code files. Add parameters
to your projects to remove hard-coded values from your code files and to perform parameter optimizations. You can add
parameters, set default parameter values, and remove parameters from your projects.
Add Parameters
Follow these steps to add an algorithm parameter to a project:
1. Open the project .
2. In the Project panel, click Add New Parameter .
3. Enter the parameter name.
The parameter name must be unique in the project.
4. Enter the default value.
5. Click Create Parameter .
To get the parameter values into your algorithm, see Get Parameters .
Set Default Parameter Values
Follow these steps to set the default value of an algorithm parameter in a project:
1. Open the project .
2. In the Project panel, hover over the algorithm parameter and then click the pencil icon that appears.
3. Enter a default value for the parameter and then click Save .
The Project panel displays the default parameter value next to the parameter name.
Delete Parameters
Follow these steps to delete an algorithm parameter in a project:
1. Open the project .
2. In the Project panel, hover over the algorithm parameter and then click the trash can icon that appears.
3. Remove the GetParameter calls that were associated with the parameter from your code files.
Get Parameters
To get the parameter values from the Project panel into your algorithm, see Get Parameters .
Number of Parameters
The cloud optimizer can optimize up to three parameters. There are several reasons for this quota. First, the optimizer only
supports the grid search strategy , which is very inefficient. This strategy tests every permutation of parameter values, so the
number of backtests that the optimization job must run explodes as you add more parameters. Second, the parameter charts
that display the optimization results are limited to three dimensions. Third, if you optimize with many variables, it increases the
likelihood of overfitting to historical data.
To optimize more than three parameters, run local optimizations with the CLI .
Optimization > Objectives
Optimization
Objectives
Introduction
An optimization objective is the performance metric that's used to compare the backtest performance of different parameter
values. The optimizer currently supports the compound annual growth rate (CAGR), drawdown, Sharpe ratio, and Probabilistic
Sharpe ratio (PSR) as optimization objectives. When the optimization job finishes, the results page displays the value of the
objective with respect to the parameter values.
CAGR
CAGR is the yearly return that would be required to generate the return over the backtest period. CAGR is calculated as
(
e
s
)
1
y − 1
where s is starting equity, e is ending equity, and y is the number of years in the backtest period. The benefit of using CAGR as
the objective is that it maximizes the return of your algorithm over the entire backtest. The drawback of using CAGR is that it
may cause your algorithm to have more volatile returns, which increases the difficulty of keeping your algorithm deployed in live
mode.
Drawdown
Drawdown is the largest peak to trough decline in your algorithm's equity curve. Drawdown is calculated as
1 −
v
t≥s min
v
smax
where v
smax
is the maximum equity value up to times and v
t≥s min
is the minimum equity value at time t where t ≥ s. The following
image illustrates how the max drawdown is calculated:
During the first highlighted period in the preceding image, the equity curve dropped from 106,027 to 93,949 (11.4%). During the
second highlighted period, the equity curve dropped from 112,848 to 99,576 (11.8%). Since 11.8% > 11.4%, the max drawdown of
the equity curve is 11.8%.
The benefit of using drawdown as the objective is that it's psychologically easier to keep an algorithm deployed in live mode if
the algorithm doesn't experience large drawdowns. The drawback of using drawdown is that it may limit the potential returns of
your algorithm.
Sharpe
The Sharpe ratio measures the excess returns relative to a benchmark, divided by the standard deviation of those returns. The
Sharpe ratio is calculated as
E[Rp − Rb
]
σp
where Rp
is the returns of your portfolio, Rb
is the returns of the benchmark, andσp
is the standard deviation of your algorithm's
excess returns. By default, Lean uses a 0% risk-free rate, so Rb = 0. The benefit of using the Sharpe ratio as the objective is that
it maximizes returns while minimizing the return volatility. It's usually psychologically easier to keep a live algorithm deployed if
it has minimal swings in equity than if it has large swings in equity. The drawback of using the Sharpe ratio is that it may limit
your potential returns in favor of a less volatile equity curve.
PSR
The PSR is the probability that the estimated Sharpe ratio of your algorithm is greater than the Sharpe ratio of the benchmark.
PSR is calculated as
P
^
SR > SR
∗ = CDF
(
^
SR − SR
∗)√n − 1
1 −
ˆγ
3
^
SR +
ˆγ
4−1
4
^
SR
2
where SR
∗ is the Sharpe ratio of the benchmark,
^
SR is the Sharpe ratio of your algorithm, n is the number of trading days,
ˆγ
3
is
the skewness of your algorithm's returns,
ˆγ
4
is the kurtosis of your algorithm's returns, and CDF is the normal cumulative
distribution function. The benefit of using the PSR as the objective is that it maximizes the probability of your algorithm's Sharpe
ratio outperforming the benchmark Sharpe ratio. The drawback of using the PSR is that, like the Sharpe ratio objective,
optimizing the PSR may limit your potential returns in favor of a less volatile equity curve.
Constraints
Constraints filter out backtests from your optimization results that do not conform to your desired range of statistical results.
Constraints consist of a target, operator, and a numerical value. For instance, you can add a constraint that the optimization
backtests must have a Sharpe ratio >= 1 to be included in the optimization results. Constraints are optional, but you can use
them to incorporate multiple objective functions into a single optimization job.
( ) (
√ )
Optimization > Strategies
Optimization
Strategies
Introduction
Optimization strategies control how the optimizer adjusts parameters for each new backtest that's run in the optimization job.
Grid search is the only strategy currently available, but you can contribute new optimization strategies.
Grid Search
Grid search is the most complete but the most expensive strategy because it takes a brute force approach and tests all the
combinations of parameter values. If you are optimizing one parameter, the grid search strategy selects the values of the
parameters based on the starting value, ending value, and step size that you provide. If you optimize two parameters, the grid
search strategy searches the Cartesian product of possible values for each parameter. The following animation shows the
process of using grid search to optimize two parameters:
In the preceding animation, grid search tests all of the parameter combinations. The axes represent the possible values of each
parameter. Gray squares represent backtests in the optimization queue, orange squares represent successful backtests, and
black squares represent failed backtests. In this example, several squares are colored at the same time because the
optimization job is using multiple optimization nodes .
Why Backtests Fail
When backtests fail during optimization, it is usually because the selected parameter values cause a divide-by-zero error or an
index out-of-range exception.
Strategy Benefit and Drawback
The benefit of the grid search strategy is that it is the most comprehensive optimization strategy. The drawback of the strategy
is it can be an expensive option because of the curse of dimensionality.
Contribute Strategies
You can contribute any optimization strategy that is popular in the literature and is not already implemented. To view the
optimization strategies that are already implemented, see our GitHub repository . If you contribute a strategy, you'll receive
some QuantConnect Credit , you'll be shown as a contributor to Lean on your GitHub profile, and your work will be used in the
Algorithm Lab by our community of over 250,000 quants.
To contribute optimization strategies, submit a pull request to the Lean GitHub repository . In your pull request, provide an
explanation of the strategy and some relevant resources so that we can add the strategy to our documentation. For an example
implementation, see the GridSearchOptimizationStrategy .
Optimization > Deployment
Optimization
Deployment
Introduction
Deploy optimization jobs for your trading algorithms to optimize your algorithm parameters for the objective that you specify.
The optimizer runs concurrent backtests to optimize your algorithm parameter using up to 24 nodes. As the optimization runs,
the results are displayed and updated in real-time.
Resources
The optimization nodes that backtest your algorithm are not the backtesting nodes in your organization. The optimization nodes
are a cluster of nodes that exclusively run optimization jobs. The optimization can concurrently run multiple backtests if you use
multiple nodes, but the maximum number of nodes you can use depends on the node type. The following table describes the
node types:
Type Description Number of Cores RAM (GB) Max Cluster Size
O2-8
Relatively simple
strategies with less
than 100 assets
2 8 6
O4-12
Strategies with less
than 500 assets and
simple universe
selections
4 12 4
O8-16 Complex strategies
and machine learning
8 16 4
The following table shows the training quotas of the optimization node types:
Type Capacity (min) Refill Rate (min/day)
O2-8 30 5
O4-12 60 10
O8-16 90 15
Cost
You can rent optimization nodes on a time basis. The deployment wizard estimates the total cost of your optimization job based
on the results of the last successful backtest of your algorithm, the number of parameters , and the optimization strategy .
Therefore, you must run a backtest of your algorithm before the deployment wizard can estimate the cost of the optimization
job. The final cost that you pay can vary from the estimate. For instance, if your backtest used parameters that were favorable
for speedy execution, the estimate can be lower than the final cost.
You can use multiple nodes to speed up the optimization job without the job costing more because you use each node for a
shorter period of time. However, there is a spin-up time of roughly 15 seconds on each backtest, so it can sometimes cost more
to use many nodes when you factor in the spin-up time. You pay for optimizations with your organization's QuantConnect Credit
balance. If you have your own hardware, you can run local optimizations with your own data and hardware.
Launch Optimization Jobs
You need the following to optimize parameters:
At least one algorithm parameter in your project .
The GetParameter method in your project.
A successful backtest of the project.
QuantConnect Credit (QCC) in your organization.
Follow these steps to optimize parameters:
1. Open the project that contains the parameters you want to optimize.
2. In the top-right corner of the IDE, click the Optimize icon.
3. On the Optimization page, in the Parameter & Constraints section, enter the name of the parameter to optimize.
The parameter name must match a parameter name in the Project panel.
4. Enter the minimum and maximum parameter values.
5. Click the gear icon next to the parameter and then enter a step size.
6. If you want to add another parameter to optimize, click Add Parameter .
You can optimize a maximum of three parameters. To optimize more parameters, run local optimizations with the CLI .
7. If you want to add optimization constraints , follow these steps:
1. Click Add Constraint .
2. Click the target field and then select a target from the drop-down menu.
3. Click the operation field and then an operation from the drop-down menu.
4. Enter a constraint value.
8. In the Estimated Number and Cost of Backtests section, click an optimization node and then select a maximum number of
nodes to use.
9. In the Strategy & Target section, click the Choose Optimization Strategy field and then select a strategy from the dropdown menu.
10. Click the Select Target field and then select a target from the drop-down menu.
The target (also known as objective) is the performance metric the optimizer uses to compare the backtest performance of
different parameter values.
11. Click Maximize or Minimize to maximize or minimize the optimization target, respectively.
12. Click Launch Optimization .
The optimization results page displays. As the optimization job runs, you can close or refresh the window without
interrupting the job because the nodes are processing on our servers.
To abort a running optimization job, in the Status panel, click Abort and then click Yes .
Optimization > Results
Optimization
Results
Introduction
The optimization results page shows your algorithm's performance with the various parameter values. Review the results page
to see how your algorithm has performed during the backtests and to investigate how you might improve your algorithm before
live trading.
View Optimization Results
The optimization results page automatically displays when you launch an optimization job . The page presents the algorithm's
equity curves, parameters, target values, server statistics, and much more information.
The content in the optimization results page updates as your optimization job executes. You can close or refresh the window
without interrupting the job because the optimization nodes process on our servers. If you close the page, you can view all of
the project's optimizations to open the page again.
Runtime Statistics
The banner at the top of the optimization results page displays the performance statistics of the optimization job.
The banner updates in real-time as the optimization job progresses on our servers. The following table describes the runtime
statistics:
Statistic Description
Completed The number of backtests that have successfully completed
Failed The number of backtests that have failed during execution
Running The number of backtests that are currently running
In Queue The number of backtests that are waiting to start
Average Length The average amount of time to complete one of the backtests
Total Runtime The total runtime of the optimization job
Total The total number of backtests run in the optimization job
Consumed The amount of QuantConnect Credit that was used to perform the optimization
Equity Curves
The optimization results page displays a Strategy Equities chart so that you can analyze the equity curves of the individual
backtests in the optimization job.
The equity curves of the backtests update in real-time as the optimization job runs. View the Strategy Equities chart to see how
the parameter values affect the equity of your algorithm, to see how sensitive the returns are to the range of parameters
selected by the optimizer, and to take a closer look at specific times in the backtest history.
Parameter Charts
The optimization results page displays parameter charts to show the relationship between the parameter value(s) selected by
the optimizer and the value of several objectives. If your optimization job has one parameter, the result page displays a scatter
plot for each objective. If your optimization job has two parameters, the result page displays a heat map for each objective.
If your optimization job has three parameters, the result page displays a 3-dimensional plot for each objective. To analyze the
results of 3-dimensional plots, you can rotate them and apply cuttoff values. To get the objective value for a combination of
parameters, hover over the dots in the plot.
Parameter Stability
Zones in the heatmap where the color of adjacent cells are relatively consistent represent areas where the objectives are stable.
In these areas, the value of the objectives is not significantly influenced by the parameter values. The following image shows
the parameter chart of an optimization job. The highlighted area identifies combinations of parameter values that stabilize the
objective function.
Supported Objectives
You can add parameter charts for the following objectives:
Alpha
Annual Standard Deviation
Annual Variance
Average Loss
Average Win
Beta
Compounding Annual Return
Drawdown
Estimated Strategy Capacity
Expectancy
Information Ratio
Loss Rate
Net Profit
Probabilistic Sharpe Ratio (PSR)
Profit-Loss Ratio
Sharpe Ratio
Total Fees
Total Trades
Tracking Error
Treynor Ratio
Win Rate
Add Parameter Charts
Follow these steps to add a parameter chart to the optimization results page:
1. In the Parameter Chart panel, click the plus icon.
2. Click the Objective field and then select an objective from the drop-down menu.
3. Click the Parameter 1 field and then select a parameter from the drop-down menu.
4. If there are multiple parameters in the optimization, click the Parameter 2 field and then select a parameter from the drop-
down menu.
5. If there are three parameters in the optimization, click the Parameter 3 field and then select a parameter from the dropdown menu.
6. Click Create Chart .
The optimization results page displays the new chart.
Individual Backtest Results
The optimization results page displays a Backtests table that includes all of the backtests that ran during the optimization job.
The table lists the parameter values of the backtests in the optimization job and their resulting values for the objectives.
Open the Backtest Results Page
To open the backtest result page of one of the backtests in the optimization job, click a backtest in the table.
Download the Table
To download the table, right-click one of the rows, and then click Export > CSV Export .
Filter the Table
Follow these steps to apply filters to the Backtests table:
1. On the right edge of the Backtests table, click Filters .
2. Click the name of the column to which you want the filter to be applied.
3. If the column you selected is numerical, click the operation field and then select one of the operations from the drop-down
menu.
4. Fill the fields below the operation you selected.
Toggle Table Columns
Follow these steps to hide and show columns in the Backtests table:
1. On the right edge of the Backtests table, click Columns .
2. Select the columns you want to include in the Backtests table and deselect the columns you want to exclude.
Sort the Table Columns
In the Backtests table, click one of the column names to sort the table by that column.
Server Stats
The optimization results page displays a Server Statistics section to show the status of the nodes running the optimization job.
The following image shows an example of the Server Statistics section:
The following table describes the information that the Server Statistics section displays:
Property Description
CPU The total CPU usage and the CPU usage of each node
RAM The total RAM usage of the RAM usage of each node
HOST The node model and the number of nodes used to run the optimization
Uptime The length of time that the optimization job has ran
View the Server Statistics section to see the amount of CPU power and RAM the optimization job demands. If your algorithm is
demanding a lot of resources, use more powerful nodes on the next optimization job or improve the efficiency of your algorithm.
Errors
The following table describes common optimization errors:
Error Description
Runtime Errors
If a backtest in your optimization job throws a runtime error, the backtest will not complete
but you will still be charged.
Data Overload
If a backtest in your optimization job produces more than 700MB of data, then Lean can't
upload the results and the optimization job appears to never be complete.
View All Optimizations
Follow these steps to view all of the optimization results of a project:
1. Open the project that contains the optimization results you want to view.
2. At the top of the IDE, click the Results icon.
A table containing all of the backtest and optimization results for the project is displayed. If there is a play icon to the left of
the name, it's a backtest result . If there is a fast-forward icon next to the name, it's an optimization result .
3. (Optional) In the top-right corner, select the Show field and then select one of the options from the drop-down menu to
filter the table by backtest or optimization results.
4. (Optional) In the bottom-right corner, click the Hide Error check box to remove backtest and optimization results from the
table that had a runtime error.
5. (Optional) Use the pagination tools at the bottom to change the page.
6. (Optional) Click a column name to sort the table by that column.
7. Click a row in the table to open the results page of that backtest or optimization.
Rename Optimizations
We give an arbitrary name (for example, "Smooth Apricot Chicken") to your optimization result files, but you can follow these
steps to rename them:
1. Hover over the optimization you want to rename and then click the pencil icon that appears.
2. Enter the new name and then press Enter .
Delete Optimizations
Hover over the optimization you want to delete and then click the trash can icon that appears to delete the optimization result.
Object Store
Object Store
Introduction
The Object Store is an organization-specific key-value storage location to save and retrieve data in QuantConnect's cache.
Similar to a dictionary or hash table, a key-value store is a storage system that saves and retrieves objects by using keys. A key
is a unique string that is associated with a single record in the key-value store and a value is an object being stored. Some
common use cases of the Object Store include the following:
Transporting data between the backtesting environment and the research environment.
Training machine learning models in the research environment before deploying them to live trading.
The Object Store is shared across the entire organization. Using the same key, you can access data across all projects in an
organization.
View Storage
The Object Store page shows all the data your organization has in the Object Store. To view the page, log in to the Algorithm Lab
and then, in the left navigation bar, click Organization > Object Store .
To view the metadata of a file (including it's path, size, and a content preview), click one of the files in the table.
Upload Files
Follow these steps to upload files to the Object Store:
1. Open the Object Store page.
2. Navigate to the directory in the Object Store where you want to upload files.
3. Click Upload .
4. Drag and drop the files you want to upload.
Alternatively, you can add data to the Object Store in an algorithm or notebook .
Download Files
Permissioned Institutional clients can build derivative data such as machine learning models and download it from the Object
Store. Contact us to unlock this feature for your account.
Follow these steps to download files and directories from the Object Store:
1. Open the Object Store page.
2. Navigate to the directory in the Object Store where you want to download files and directories.
3. Select the file(s) and directory(ies) to download
4. Click Download .
5. Wait while QuantConnect processes the request.
6. Click the Download link that appears.
Storage Sizes
All organizations get 50 MB of free storage in the Object Store. Paid organizations can subscribe to more storage space. The
following table shows the cost of the supported storage sizes:
Storage Size (GB) Storage Files (-) Monthly Cost ($)
0.05 1,000 0
2 20,000 10
5 50,000 20
10 100,000 50
50 500,000 100
Delete Storage
Follow these steps to delete storage from the Object Store:
1. Open the Object Store page.
2. Navigate to the directory in the Object Store where you want to delete files.
3. Click the check box next to the files you want to delete.
4. Click Actions and then click Delete from the drop-down menu.
5. Click OK .
Alternatively, you can delete storage in an algorithm or notebook .
Edit Storage Plan
You need storage billing permissions and a paid organization to edit the size of the organization's Object Store.
Follow these steps to edit the amount of storage available in your organization's Object Store:
1. Log in to the Algorithm Lab.
2. In the left navigation bar, click Organization > Resources .
3. On the Resources page, scroll down to the Storage Resources and then click Add Object Store Capacity .
4. On the Pricing page, select a storage plan.
5. Click Proceed to Checkout .
Research to Live Considerations
When you deploy a live algorithm, you can access the data within minutes of modifying the Object Store. Ensure your algorithm
is able to handle a changing dataset.
The live environment's access to the Object Store is much slower than in research and backtesting. Limit the individual objects
to less than 50 MB to prevent live trading access issues.
Usage by Project
The Resources page shows the total storage used in your organization and the storage used by individual projects so that you
can easily manage your storage space. To view the page, log in to the Algorithm Lab and then, in the left navigation bar, click
Organization > Resources .
Community
Community
The QuantConnect community consists of over 250,000 quants and investors with diverse backgrounds. Our platform supports
several channels of communication so that our members can discuss with the core team, other community members, and thirdparty contractors. Our community members are a great source for assistance when creating trading algorithms, but we
recommend all users complete our base training material before requesting assistance from other members.
Code of Conduct
Community policies and expectations
Forum
Discuss with other members
Discord
Outside of our forum
Profile
Your public profile page
Quant League
Open-source student competition
Quant League Pro
Connecting Authors and Funds
Academic Grants
Funding for researchers
Integration Partners
Quant professionals for hire
Affiliates
Monetize your platform
Research
Publish formal strategy research
See Also
Using the Forum
Managing Your Profile
Community > Code of Conduct
Community
Code of Conduct
Introduction
The QuantConnect community consists of 250,000 quants and investors with diverse backgrounds. Our platform supports
several channels of communication so that our members can discuss with the core team, other community members, and thirdparty contractors. Our community members are a great source for assistance when creating trading algorithms, but we
recommend all users complete our base training material before requesting assistance from other members.
Our community forum was created to be a hub for sharing quality quantitative science insights, discussions on quantitative
philosophies, and solving problems. Our goal is to embrace new ways of thinking while keeping a friendly, welcoming
environment where people feel comfortable amongst their peers. Our community's exceptional and diverse range of opinions
and experiences make us a unique platform. Coming together with mutual respect and courtesy can unite us in growing together
as quants.
We ask that our users adhere to the community code of conduct to ensure QuantConnect remains a safe, healthy environment
for high-quality quantitative trading discussions.
Expectations
If you're here to get help, make it as easy as possible for others to help you. Follow our guidelines and remember that our
community is made possible by volunteers.
Be clear and constructive when giving feedback, and be open when receiving it. Edits, comments, and suggestions are healthy
parts of our community.
If you're here to help others, be patient and welcoming. Learning how to participate in our community can be hard. Offer support
if you see someone struggling or otherwise in need of help.
Be inclusive and respectful. Avoid sarcasm and be careful with jokes — tone is hard to decipher online. Prefer gender-neutral
language when uncertain. If a situation makes it hard to be friendly, stop participating and move on.
The following table shows examples of friendly and unfriendly content:
Unfriendly Friendly
"Google is free!"
"I think googling this might provide you with more helpful
information."
"Obviously that's wrong because..." "I think I can help you with this! Try this..."
"Can you speak English?" "I think you're trying to say ____. Is that correct?"
"Your strategy will never work because ___." "Here are some suggestions for your strategy..."
Policies
Please follow our community policies.
Respect
We want our forum to be a place of general respect for one another. Keep interactions constructive, but friendly and
lighthearted. Remember — we're all here to help one another and keep our community strong.
Due Diligence
We have hundreds of quants posting their questions. Your question is important, but make sure to check a few places before
posting. We've worked hard on providing comprehensive documentation and bootcamp tutorials — make sure to check there
first. Next, try Googling the concept to see if you can get another perspective before posting. Furthermore, our Debugger is a
great tool to identify bugs in the code logic.
Relevancy
Keep posts related to algorithmic trading or quantitative finance.
Patience
Be patient with the community responses to your questions. Keep in mind the community are volunteers contributing to your
quantitative growth voluntarily. When possible, answer your own questions to leave a path for future readers. Avoid "bumps",
"+1", "Any Update?", double posting, thread hijacking , or necro-bumping discussions. Contributions are often rewarded with
QuantConnect Credit .
Bigotry
QuantConnect is firmly rooted in our policy against bigotry in our community forum. We are vehemently against racist, sexist,
xenophobic, homophobic, or otherwise discriminatory behavior in our community. Any language that may offend anyone based
on race, sexual orientation, gender, religion, will not be tolerated.
Harassment
Bullying is not a part of our core culture at QuantConnect. We do not tolerate bullying, sexual harassment, profanity, threats of
violence or otherwise, or any sexual harassment in our community forum.
Bug Reports / Data Issues
We request bug reports be sent to the Support Team . The forums are not an effective bug tracking tool and often the reported
issue is simply confusion on how the platform operates. For data issues please report the specific dates, times, contracts, and
type of issue to the Data Explorer Issues List . This is a system we've designed to track, fix and notify users when issues are
fixed.
Promotional Activity
Spam and other forms of promotional activity isn't permitted in the community forum. Posts that deliver immediate value to the
readers are permitted such as sharing a well-performing algorithm with an attribution to the author's company in the code
comments.
Can Someone Make My Algorithm?
We understand people are at different stages of their quant-growth, but if you are soliciting assistance you should have
completed Boot Camp , and attach a backtest or code snippet with your best attempt at building your algorithm. This shows
respect for the reader's time. You will need to know how to code to use QuantConnect.
Reporting Violations
We monitor our forum diligently, but if you see something unsavory, please message us to report the activity.
About This Code of Conduct
We aspire to have a welcoming community filled with high-quality discussions about quantitative and algorithmic trading. Since
our founding in November 2012, we ran an informal code of conduct and evolved those principles to support the community.
Starting January 1st, 2021 as the scale of community content surpassed our ability to review each post, we have sought to write
down these guidelines to provide transparency and framework for productive discussions. We welcome your feedback and
expect this code of conduct to evolve over time.
Community > Forum
Community
Forum
Introduction
The QuantConnect forum is a place to discuss with other community members, Mia (our AI assistant), the core QuantConnect
team, and our Integration Partners . Use the forum to spark interesting discussions, ask for assistance from other members, and
voice your opinion on our products. You must complete 30% of the Bootcamp lessons in the Learning Center to unlock the
ability to post to the forum.
Discussions
Discussions are a set of forum posts and comments about a targeted subject. We occasionally post to forum discussions to
announce new features, tutorials, and examples. If you have completed 30% of the Bootcamp lessons, you can contribute to
discussions. Create discussions to connect with other members or to ask for guidance on a specific problem that you are facing,
but always perform your own research and tests before asking other members for assistance.
The forum supports the following functionality:
Attaching backtest results and research notebooks
Bookmarking discussions
Generating links to comments to share
Publishing text content that includes bold, italics, lists, links, emojis, and LaTeX
Rewarding discussion content with QuantConnect Credit and upvotes
Sharing code snippets, photos, videos, and files
Tagging other members
Sharing Backtests
You can attach project files and backtest results to your forum comments. Sharing your project files is helpful in the following
situations:
Contributing to a discussion about trading strategies and implementations
Highlighting an issue with data or Lean
Requesting assistance with an algorithm or research notebook
Sharing your interesting research and backtests with the community
Sharing backtests is helpful in these situations because it enables other members to reproduce your results. However, since
Lean is under constant development, old backtests that are attached to forum discussions can sometimes produce different
results than they did in the past.
You can only attach backtests to the forum if they don't throw errors. If you want to share a backtest that throws errors, call the
quit method method before the error occurs.
Credit
To show your appreciation for contributions in the forum, give some QuantConnect Credit (QCC) rewards . The following table
shows the available QCC rewards:
Award Description Cost (QCC)
Silver Award
A simple token of recognition from one
quant to another. Keep up the great
work.
80
Gold Award
This is some great work! Gold star! 600
Platinum Award
Highly resistant to oxidation, this
award is for those contributions which
will stand the test of time. Strong,
classic, and useful for most high
technology products.
1,200
Medal for Excellence
The QuantConnect Medal for
Excellence is awarded by a member of
the QuantConnect staff for
exceptional contributions to the
QuantConnect community.
3,000
Plutonium Award
Nuclear hot! This post is incredible
and deserves recognition as such.
Show the author your appreciation for
their work.
2,000
Docs Shakespeare
You've left a mark by a contribution to
the documentation for the community.
Your edits and examples will be
followed for generations to come.
500
Nobel Laureate
Bestowed in recognition of
quantitative advances.
80
Spaghetti Code Award
Following the intricate flows of code
noodle, this code compiles and runs.
300
Jedi Quant
Quant promise you show. Your code
channel the force into. Hmmmmmm.
200
Live Trader
That looks like a wild ride. 50
Today I Learned
Thank you for upgrading my brain. 50
Machine Unlearning
I plan on letting the computers do all
the work for me.
150
Totally Overfit
I see parameters everywhere. 80
Mind Blown Award
Something incredibly amazing, mindboggling, and you're shocked
senseless.
80
Research Rembrandt
A Jupyter notebook work of art,
pulling together all the right hues and
plots to be a true masterpiece.
100
Cayman Island Award
Let's get a boat and start a fund
together. Did I mention the boat?
800
Printing Money Award
Awarded to profitable algorithms. 100
Stronger Together Award
We're stronger together. Let's make
this happen!
120
Appreciate the Support Award
Quant trading is a hard road, we could
all use a hand.
250
Master Craftsman
You're a master craftsman, taking raw
materials and molding them into works
of art for the good of the world.
600
Notifications
Follow discussions to receive email notifications when members post new content. You can follow all forum discussions or
individual discussions.
All Discussions
Follow these steps to subscribe or unsubscribe from all forum discussions:
1. Log in to your account.
2. Open the community feed .
3. In the Community Mailing List section, click Subscribe or Manage Subscriptions .
Individual Discussions
To toggle your subscription to individual discussions, add or remove a bookmark to the discussions.
Search Discussions
Follow these steps to search the forum:
1. Open the forum homepage .
2. Click the magnifying glass icon.
3. Enter some keywords to use in the search.
Create Discussions
Follow these steps to create a new forum discussion:
1. Log in to your account.
2. Open the forum homepage .
3. Click Start New Discussion .
4. Enter the discussion title.
5. Enter the discussion content.
6. (Optional) Follow these steps to attach a backtest:
1. Click Attach Backtest .
2. Click the Project field and then select the project that contains the backtest that you want to attach from the dropdown menu.
You can only attach projects that you own.
3. Click the Backtest field and then select a backtest from the drop-down menu.
When you attach a backtest, you're sharing all of the project files that were used to generate the backtest results.
7. (Optional) Under the Discussion Tags section, click Show All and then click all of the tags that are relevant to the
discussion.
8. Click Publish Discussion .
Post Comments
You can only post comments on open forum discussions. To open a closed discussion, contact us .
Follow these steps to post a comment on a forum discussion:
1. Log in to your account.
2. Open the forum homepage .
3. Click the discussion on which you want to comment.
4. If you want to reply to a comment in the discussion, click Reply at the bottom of the comment.
5. If you want to comment on the top-level discussion, scroll to the bottom of the discussion page and enter your comment.
6. (Optional) If you comment on the top-level discussion, follow these steps to attach a backtest:
1. Click Attach Backtest .
2. Click the Project field and then select the project that contains the backtest you want to attach from the drop-down
menu.
You can only attach projects that you own.
3. Click the Backtest field and then select a backtest from the drop-down menu.
When you attach a backtest, you're sharing all of the project files that were used to generate the backtest results.
7. Click Reply .
Your comment is appended to the discussion.
Accept Answers
You can only accept answers on discussions you create.
Follow these steps to accept an answer on your discussion:
1. Log in to your account.
2. Open the forum homepage .
3. Open the discussion that contains the answer that you want to accept.
4. On the comment that you want to accept, click Accept Answer .
The discussion closes and a copy of the accepted answer is added as the first comment in the discussion.
Give Rewards
You need sufficient QuantConnect Credit (QCC) in your organization to give QCC rewards.
Follow these steps to reward a discussion or comment:
1. Log in to your account.
2. Open the forum homepage .
3. Open the discussion to which you want to give your reward.
4. If you want to give a QCC reward, follow these steps:
1. On the discussion page, if you want to reward the discussion, click Reward Discussion . If you want to reward a
comment in the discussion, on the comment that you want to reward, click Reward Answer .
2. Click a QCC reward.
3. Click Send Reward .
The reward displays and the QCC value of the reward is deducted from your organization's balance.
5. If you want to give an upvote, click the thumbs-up icon on the comment that you want to reward.
Share Comments
Follow these steps to share a link to a comment in the forum:
1. Open the forum homepage .
2. Click the discussion that contains the comment that you want to share.
3. On the discussion page, in the bottom-right corner of the comment that you want to share, click the share icon and then
click one of the share options.
Edit Comments
You can edit comments only within five minutes after publishing the comments.
Follow these steps to edit a comment:
1. Log in to your account.
2. Open the forum homepage .
3. Click the discussion that contains the comment that you want to edit.
4. On the discussion page, in the top-right corner of the comment that you want to edit, click the three dots icon and then
click Edit .
5. (Optional) Update the comment text.
6. (Optional) Follow these steps to update the attached backtest:
1. Click Update Backtest .
2. Click the Project field and then select the project that contains the backtest that you want to attach from the dropdown menu.
You can only attach projects that you own.
3. Click the Backtest field and then select a backtest from the drop-down menu.
7. Click Update .
The updated comment displays.
Delete Comments
Email us the comment URL and a reason for deleting the comment. We will consider deleting the comment.
Access Member Profiles
Open the forum homepage , click a discussion, and then click a member's profile image to view the member's profile page.
Manage Bookmarks
Follow these steps to bookmark a discussion:
1. Log in to your account.
2. Open the forum homepage .
3. Click on the discussion that you want to bookmark.
4. On the discussion page, in the bottom-right corner of the first comment, click the bookmark icon.
Click the bookmark icon again to remove the discussion from your bookmarks.
On the forum homepage, click Bookmarked to view all of the discussions that you have bookmarked.
Community > Discord
Community
Discord
Introduction
Join the QuantConnect Discord server to chat with community members, Mia (our AI assistant), and the core QuantConnect
team in real-time.
Code of Conduct
The Discord server is governed by the normal rules in the Code of Conduct .
Community > Profile
Community
Profile
Introduction
Your QuantConnect profile page is your place to customize your community appearance. Your profile page is accessible to
everyone from the forum. The page displays your personal information, your preferred programming language, your latest
forum activity, and the badges that you have earned.
Personal Information
Your profile page displays the following personal information about you:
Username
Title
Picture
Biography
Social media accounts
Website
The date that you joined QuantConnect
You can edit your profile image, username, title, biography, social media links, and email.
Profile Image
Follow these steps to update your profile image:
1. Log in to your account.
2. In the top navigation, click yourUsername > Change Profile Image .
3. Click Browse .
4. Select a file from your local machine and then click Open .
Your profile image must be in gif , jpg , or png format and less than 1MB in size.
5. Click Save .
"Profile image changed successfully" displays.
Username
Follow these steps to update your username:
1. Log in to your account.
2. In the top navigation, click yourUsername > My Profile .
3. On your profile page, hover over your username and then click the Edit box that appears.
4. Enter your new username and then click OK .
Your new username displays.
Title
Follow these steps to update your title:
1. Log in to your account.
2. In the top navigation, click yourUsername > My Profile .
3. On your profile page, hover over your title and then click the Edit box that appears.
4. Enter your new title and then click OK .
Your new title displays.
Biography
Follow these steps to update your biography:
1. Log in to your account.
2. In the top navigation, click yourUsername > My Profile .
3. On your profile page, hover over your biography and then click the Edit box that appears.
4. Enter your biography and then click OK .
Your new biography displays.
Social Media Links
Follow these steps to update your social media links:
1. Log in to your account.
2. In the top navigation, click yourUsername > My Profile .
3. On your profile page, hover over the social media icon that you want to update and then click the Edit box that appears.
4. Enter your social media link and then click OK .
Email
Email us to change the email associated with your QuantConnect account.
Organizations
Your profile page displays all of the organizations of which you are a member and the date that you joined each organization.
Activity
Your profile page displays your last five forum posts and your community engagement statistics. The forum post display lets
other members read your recent posts and lets you find the comments that you recently posted. The page displays your
community engagement statistics like the number of experience points that you've earned, the number of comments that
you've posted, the number of discussions that you've created, and the number of algorithms that you've shared in the forum.
Your community engagement statistics are displayed so that other members can see how active you are in the forum.
In addition to your forum activity, your profile page also platform engagement statistics like the number of backtests that you've
run and the proportion of the Bootcamp lessons that you've completed. Your platform engagement statistics are displayed so
that other members can see how experienced you are with QuantConnect and Lean.
Badges
Badges are fun to collect and show your experience with QuantConnect. Your profile page displays the badges you've earned.
The following table describes the badges you can earn:
Badge Description
Algorithm Sharing
Share an algorithm with the community.
Chatter Box
Post 100+ comments in the forum.
Backtester
Run 100+ backtests.
Live Trading
Deploy a live trading algorithm.
Parallel Live Trading
Run multiple live trading algorithms simultaneously.
Connected With Discord
Connect your QuantConnect account with our Discord
server .
C# Programming
Set C# as your preferred programming language.
Python Programming
Set Python as your preferred programming language.
Popular
Receive 10+ upvotes from different members.
Boot Camp Private
Finish the easy tutorial.
Boot Camp Corporal
Finish the intermediate tutorial.
Boot Camp Major
Finish the advanced tutorial.
Notebook Sharing
Attach a research notebook to a post in the community
forum.
Broadcaster
Attach a live running algorithm to a forum discussion.
Cloner
Clone 100+ algorithms from the community forum.
Bug Hunter
Report 10+ bugs.
Open Source Contributor
Make a pull request to one of our GitHub repositories .
Alpha Streams Developer
Add 1 Alpha to the Alpha Streams Market.
Alpha Streams Artisan
Add 3 Alphas to the Alpha Streams Market.
Alpha Streams Master
Add 10 Alphas to the Alpha Streams Market.
Alpha Winner #1
Place first in the Alpha competition.
Alpha Winner #2
Place second in the Alpha competition.
Alpha Winner #3
Place third in the Alpha competition.
Community Editor
Provide great service to community members in the forum
or the Discord server.
Outstanding Community Service
Provide excellent service to community members in the
forum or the Discord server.
Log In
Follow these steps to log in to your account:
1. In the top navigation bar, click Sign In .
The Sign In to QuantConnect page displays.
2. If you signed up with your Google or Facebook account, click Sign In with Google or Sign In with Facebook and then follow
the prompts.
3. If you signed up with your email address, enter your email address, enter your password, and then click Sign In .
4. If you have 2FA enabled on your account, in the Please Enter 2FA Code field, enter the authentication code from the
Google Authenticator app on your mobile device and then click Sign In .
Toggle 2FA
Follow these steps to activate or deactivate two-factor authentication (2FA):
1. Log in to your account.
2. In the top navigation bar, click yourUsername > My Account .
3. If you want to activate 2FA, follow these steps:
1. On your Account page, in the Security section, click Activate .
2. Enter your email address and then click Continue .
3. Follow the steps in the pop-up window and then click Activate Two Factor .
"Updated successfully" displays.
4. If you want to deactivate 2FA, follow these steps:
1. On your Account page, in the Security section, click Deactivate .
2. Enter your email address and then click Continue .
"Success" displays.
Request API Token
Follow these steps to request an API token:
1. Log in to your account.
2. In the top navigation bar, click yourUsername > My Account .
3. On your Account page , in the Security section, click Request Email With Token and Your User-Id for API Requests .
4. Click OK .
We email you your user Id and API token.
Reset API Token
Follow these steps to reset your API token:
1. Log in to your account.
2. Stop all the backtests, optimization jobs, live algorithms, and notebooks you're running.
3. In the top navigation bar, click yourUsername > My Account .
4. On your Account page, in the Security section, click Reset My Token .
5. Click OK .
6. Check your email to get your new API token.
Close Active Sessions
Follow these steps to close active sessions:
1. Log in to your account.
2. In the top navigation bar, click yourUsername > My Account .
3. On your Account page, in the Security section, click Sign Out next to the session that you want to close.
"Session terminated successfully" displays.
Manage Email Subscriptions
Follow these steps to manage your email subscriptions:
1. Log in to your account.
2. In the top navigation bar, click yourUsername > My Account .
3. On your Account page, in the Account Settings section, click Manage Email Subscriptions .
4. Select the subscriptions you want and deselect the subscriptions you don't want.
"Preferences updated successfully" displays.
Change Password
Follow these steps to change your password:
1. Log in to your account.
2. In the top navigation bar, click yourUsername > Change Password .
3. Enter your current password.
4. Enter your new password.
5. Enter your new password again to confirm.
6. Press Reset Password .
7. Enter your email address and then click Continue .
Sign Out
In the top navigation bar, click yourUsername > Sign Out to sign out.
Deactivate Account
Follow these steps to deactivate your account:
1. Log in to your account.
2. In the top navigation bar, click yourUsername > My Account .
3. At the bottom of your Account page, click Deactivate Account .
4. Click Deactivate Account .
5. Enter your email address and then click Continue .
The QuantConnect homepage displays. Log in to reactivate your account.
Remove Account
Contact us to remove your QuantConnect account from our servers.
Community > Quant League
Community
Quant League
Introduction
Every year, we see thousands of applications for internships from driven and intelligent students seeking opportunities to
springboard their careers. The top industry funds take less than 1% of applicants, leaving thousands of students unable to prove
their abilities.
The Open Quant League is an opportunity for university classes and clubs to showcase their quantitative trading skills. Team
members do research publicly and deploy algorithms live, gaining a track-record. All teams that enter the competition receive
the following benefits:
Free credit to use QuantConnect.
A platform to discuss their strategy logic with a public, permanent link to share on their LinkedIn profiles and resumes.
Inbound inquiries from employers in the QC community looking for help on their algorithms or looking to hire talent for their
fund.
Public quarterly rankings page to share with potential employers via URL, including your discussion thread.
In addition, all QC community members benefit from gaining access to the research from each university that participates. The
live algorithm the team contributed will be public, so community members can use it to learn, trade, and improve.
League Comparison
The following table summarizes the differences between Quant League and Quant League Pro :
Quant League Quant League Pro
Live track record Paper trading Real money
Team size 3+ students 1+ trader
Code visibility Open source Closed source
Author visibility
Public listing shows the Authorʼs
username and LinkedIn profile
Private listing with an anonymous
Author
Minimum account size N/A $30,000 USD
Minimum capacity N/A
$1M minimum capacity during
backtests
Ranking Sharpe Ratio Sharpe Ratio
Prizes Yes Yes
Schedule
The competition runs on a quarterly schedule. At the end of each quarter, the competition resets and the algorithms in the
current competition are automatically resubmitted to the next competition.
Academic Scholarships
For every team that participates, QuantConnect provides Trading Firm seats with Bronze-tier support and an increased number
of backtest, research, and live trading nodes for a total value of $20,000 per year. Competitors can use these perks without a
cost for their team's submission and personal projects.
Note: The scholarship will be revoked for teams who don't submit a QuantLeague entry.
Rankings
Algorithms are simply ranked by net performance over the quarter, but this condition is subject to change. To view current
rankings, see League .
Prizes
The top three organizations at the end of the quarterly competition receive prices of 500, 300, and $200, respectively. We are
looking to increase these prizes via Corporate Sponsorships .
Additionally, the members of the winning team will get a chance to interview for a QuantConnect internship.
Enter the Competition
To enter your team into the competition, the team captain should submit the following information into this Google Form :
Team's University
Team name
The email with which the captain is registered on the QuantConnect platform
The emails with which the team members are registered on the QuantConnect platform
A confirmation that all team members have updated their Names, Profile Picture, and LinkedIn Profiles on the
QuantConnect platform
The link to the backtest of the team's submission
The team's investment thesis
For the investment thesis, introduce your strategy concepts and philosophy to help investors and funds understand your
investment idea. Communicating your thesis and attracting investment for your projects is critical to quantitative finance. You
can encourage discussion and contributions by posting interesting ideas on your team's page.
Your team must observe the following rules:
Your team has 3-15 members.
All members are proficient at coding with the QuantConnect API .
All members have completed Boot Camp .
All members have updated their Name, Profile Picture, and LinkedIn account on the platform so funds in the QuantConnect
ecosystem can offer you placements.
Your algorithm must observe the following rules:
Your algorithm is unique. Teams are penalized for submitting algorithms that are nearly identical to other competitors.
Your algorithm is invested at all times.
Your algorithm trades liquid securities . For Equities, this includes the top 1000 stocks, Crypto includes the top 10 coins,
Forex includes the top 20 pairs, and Futures includes the top 30 contracts.
The maximum leverage is 2.
Once you submit the form, the QuantConnect team will review your submission and list it to the Quant League.
Update Submissions
Follow these steps to update your team's algorithm:
1. Log in to your account .
2. On the Open Quant League page, scroll down to find your team's submission, and click View .
3. On the right side of the page, click Update Competition .
4. On the Update League Submission page, enter the algorithm name and description into the form.
5. Follow these steps to attach a backtest:
1. Click Attach Backtest .
2. Click the Project field and then select the project that contains the backtest that you want to attach from the dropdown menu.
Note: You can only attach projects that you own.
3. Click the Backtest field and then select a backtest from the drop-down menu.
6. Click Update League Submission .
Your algorithm has been updated and is competing in the league. Congratulations!
Intellectual Property
The source code of each algorithm is private during its first quarter in the competition. At the end of its first quarter in the
competition, the source code will be visible, even if the algorithm is submitted to the next competition. To keep your source
code private, submit a new algorithm for each quarter.
Remember, the goal is to demonstrate the strength of your abilities which is easier when employers can review the code you're
writing.
Quotas
Schools and clubs can have up to two entries in the competition each quarter, one for an academic course and one for a student
society. For example, Duke University can have one entry for FECON 413 and another entry for the Duke Investment Club.
Corporate Sponsorships
We are proud to share the community's support for Quant League's university talent with the following three 3 sponsorship
tiers. We direct all the funds we receive from the sponsorship to increasing student prizes.
Benefit Bronze Tier Silver Tier Gold Tier
Student Resume Book
Email Access to the
Participants
Company link featured on
Quant League website
Company logo placement
on all Quant League
materials
Factor reports of every
competing submission
Live Signal Exports from all
submissions
Personalized prompts for
strategy submissions
Price (USD per quarter) $1,250 $5,000 $7,500
With our sponsorship tiers starting at just $1,250 USD, we would love for you to join us in making Quant League a tremendous
success! To make the contribution and support student talent, contact our team.
Pamphlet
To share information about this competition with your team members or university, feel free to download and distribute the
following pamphlet.
Common Mistakes
The following video explains some common mistakes teams make in their submission algorithms:
Clone the algorithm code .
Community > Quant League Pro
Community
Quant League Pro
DRAFT DOCUMENTATION: Please note this is draft documentation and subject to change. We aim to release a beta of Quant
League Pro June 10th.
Introduction
Quant League Pro is a quarterly competition that strives to connect quant traders who have live real-money track records with
third-party institutional funds that are looking to allocate to trading strategies. Authors perform research and submit algorithms
to the competition. The quarterly performance of all strategies in the competition is publicly available for Funds. When a Fund
allocates to a strategy, the Fund creates a Separately Managed Account (SMA) through their brokerage and provides the
credentials to the strategy Author. The Author then deploys the algorithm on QuantConnect Cloud with the SMA credentials and
manages the deployment by placing manual trades and updating the algorithm when necessary.
The strategies competing in Quant League Pro run with real money from the strategy Author. In addition to the returns they
generate by running their own strategy and the earnings from their allocation agreement with the Fund, Authors can earn
quarterly prizes if their strategies rank near the top of the leaderboard.
League Comparison
The following table summarizes the differences between Quant League and Quant League Pro:
Quant League Quant League Pro
Live track record Paper trading Real money
Team size 3+ students 1+ trader
Code visibility Open source Closed source
Author visibility
Public listing shows the Authorʼs
username and LinkedIn profile
Private listing with an anonymous
Author
Minimum account size N/A $30,000 USD
Minimum capacity N/A
$1M minimum capacity during
backtests
Ranking Sharpe Ratio Sharpe Ratio
Prizes Yes Yes
Schedule
The competition runs on a quarterly schedule. At the end of each quarter, the competition resets and the algorithms in the
current competition are automatically resubmitted to the next competition.
Rankings
Algorithms are simply ranked by their Sharpe ratio over the quarter, but this condition is subject to change. To view current
rankings, see _____.
Prizes
At the end of the quarterly competition, the top three algorithms receive prizes of 500, 300, and $200, respectively. Additionally,
all participants receive QuantConnect Credit for their live node hosting cost up to $48 per month.
Enter the Competition
To add an algorithm to the competition, follow these steps:
1. ______
2. ______
3. ______
You must observe the following rules:
You must be proficient at coding with the QuantConnect API .
You must have completed Boot Camp .
Cash deposits during the quarter result in disqualification.
Your algorithm must observe the following rules:
Your algorithm is unique.
Your algorithm is actively trading.
Your algorithm trades liquid securities . For Equities, this includes the top 1,000 stocks, Crypto includes the top 10 coins,
Forex includes the top 20 pairs, and Futures includes the top 30 contracts.
The maximum leverage is 2.
Update Submissions
Follow these steps to update your Quant League Pro algorithm:
1. Stop the algorithm .
2. Update the code.
3. Re-deploy the algorithm on QuantConnect Cloud.
Quotas
Authors can have up to three entries in the competition.
Fund Requirements
Members in organizations on the Institution tier can contact the Author of Quant League Pro algorithms. Funds must agree to the
following terms:
QuantConnect should be copied in correspondence with the strategy Author.
Allocations should be done through algorithms hosted on the QuantConnect platform.
Authors should be permitted to keep their strategy publicly listed.
These conditions are to continue the viability of Quant League Pro listings. This service is only open to commercial entities that
pass accredited investor status. They must have a valid Institutional account on QuantConnect.
To allocate to a strategy, follow these steps:
1. Book a meeting with our team to ensure it's the right fit for your firm.
2. Create a Separately Managed Account in your brokerage account and assign permission to the Author to access it.
3. The Author manages the allocation with his closed-source project code.
Allocation Agreements
Contract negotiations are between the Author and the Fund, and QuantConnect is not involved. QuantConnect serves as a
platform for discovering talented quants in our community and facilitating introductions.
Common allocation agreements include intellectual property, revenue share, high water marks , and the allocation schedule. It's
common to start with a small allocation (for example, $500K) and scale up as the strategy proves itself in live mode. For an
example agreement, see the Example Allocation Agreement .
Community > Academic Grants
Community
Academic Grants
Introduction
The Academic Grant program gives researchers publishing a paper up to a three-month subscription. With this grant, you can
implement your strategy and generate public source code that anyone can use to reproduce your results with our open-source
platform on a uniform dataset. You can use the grant to create a Team organization with multiple members collaborating on the
research.
Apply for Grants
To be eligible for this grant, you must meet the following criteria:
1. You must be a member of an academic or corporate institution with a track record of published papers in peer-reviewed
journals of the quantitative finance field.
2. You must prove that you are a member of the institution as a professor, researcher, or student.
3. You must be actively doing research with the goal of writing a research paper.
Your research should be achievable using the QuantConnect platform and data. For example, we don't currently accept
research on bonds, stock warrants, and non-US stocks. You can only use external data sources from official government
websites, like FRED , or data vendors with a good track record. To view the dataset currently available on QuantConnect, see
the Dataset Market .
To apply for a grant, submit your research proposal based on a draft paper or thesis to be published to
research@quantconnect.com , along with details of your academic program or corporate research program.
Republishing Permission
By accepting a grant, you agree to share your research with a write-up, including the code, with the QuantConnect Research
forum . If applicable, your strategy may be deployed as an example (with full credit given) on the Explorer page .
Community > Integration Partners
Community
Integration Partners
Introduction
Our Integration Partners are hand-picked, independent consultants and companies with a solid track record of operational
excellence with QuantConnect. They offer guidance, consultation, teaching, coding development services. Let them help take
your idea to a fully implemented algorithmic trading strategy.
Hire Integration Partners
The Integration Partners are thoroughly vetted and must pass a test to provide their services to the community. They decide
their pricing and the services that they provide. Their services range from beginner to advanced dives into the algorithm
framework, strategy and project consultation, developing simple to complex algorithms, portfolio optimization, and more. Hire
them on the Integration Partners page to connect with a QuantConnect expert, to progress your development skills, or to
outsource your algorithm development.
Join the Integration Partner Team
To list your services on the Integration Partners page, contact us and pass our test. If you are accepted and community
members hire you, you receive 100% of the revenue. As an Integration Partner, you set your own pricing, define your own
services, pick your own hours, and get paid to work on your quant trading specialty. Our Integration Partner program is a great
addition to resumes.
Community > Affiliates
Community
Affiliates
Introduction
Our Affiliate Program gives influential quants, financial strategists, and traders an opportunity to use their platforms to share
QuantConnect and subsequently share in profits from referrals.
Earning Potential
On a monthly basis, you'll receive 10% of sales that originate for your referral. You'll receive this for up to 12 months after the
individual you refer subscribes to our services. For example, if you refer a customer who signs up for our
280/monthplanandthecustomstayswithusfor12 + months, you
′
llreceive12paymentsof28 for a total of $336 over the course of the first
year. If the customer leaves after five months, you'll receive compensation for those five months.
YouTube Case Study
Chris is a YouTube host with an audience of more than 20,000 subscribers where he posts trading and finance-related content
in an easy-to-digest format. He is often appreciated for his ability to simply present complex concepts. Chris originally
organically posted a review of various algorithmic trading platforms that caught our eye, so we invited him to join the
QuantConnect Affiliate Program.
Over a period of two months, Chris and the QuantConnect team collaborated to design a series of video topics that would be
interesting for the broader community. Chris always had complete content ownership and artistic control over his content. He
often sought a “content reviewˮ from the QuantConnect team in a private Slack channel to help maintain the technical accuracy
and quality of the videos. We were mindful to update him about new features that might impact or improve his videos. Each time
Chris published a video, we would re-share it on all our social media channels to increase his video reach. We were always
motivated to get high-quality content to the community so they can learn more effectively.
Through his channel and the partnership with QuantConnect, Chris was able to create a long-term passive income from his
content. Each month, he drives approximately 500 users to QuantConnect and 10-15 subscribe to become long-term clients.
Within six months, he has built a passive recurring revenue stream of $250/month.
Become a Partner
To apply for the QuantConnect Affiliate Program, fill in the application form .
Community > Research
Community
Research
Introduction
The Research page contains articles from QuantConnect team members and community members that implement a particular
trading strategy. Review these articles to gain a better understanding of creating full trading algorithms with LEAN and the
method to build them. If you have an interesting strategy or research you want to share with the community, create a post and
get featured on the Research page and distributed in the community mailing list. Sharing research can be a great way to build a
reputation within the QuantConnect community, reach potential employers, and earn QuantConnect Credit .
Submit Proposals
Before you investigate a trading idea, submit a proposal. The proposal should describe the type of strategy you want to
investigate, the universe and asset classes it's applied to, the datasets it requires, results you expect to see, and any research
papers you're using as a source. Once you submit a proposal, we will review it and email you on whether it's approved or
rejected.
We only approve research that is based on some financial concept, theory, or model. We reject research that is just a
combination of technical indicators and overfit parameters. This approval process ensures you avoid spending time on research
that won't end up being published.
Follow these steps to submit a research proposal for the Research page:
1. Open the Research page.
2. Click Share New Research .
3. In the Title field, enter the title of your research.
The title must follow standard capitalization rules. For example, "Opening Range Breakout for Stocks in Play".
4. In the Content field, replace the placeholder text in the Introduction section of the template with the introduction of your
submission.
During the proposal stage, the Introduction section should summarize your area of research. Explain what type of strategy
it is, the universe and asset classes it's applied to, the datasets it requires, and results you expect to see. If the strategy is
based on a research paper, reference the paper at the end of the introduction.
5. If you have references, list them in the References section.
6. Click Publish Research .
Publish Content
After you receive our email that your research proposal is approved, implement your strategy or research notebook and then
follow these steps to add your findings to your research post:
1. Open the Research page.
2. Click on the draft of your research post.
3. On the discussion page that opens, in the top-right corner of the Introduction section, click the three dots icon and then
click Edit .
4. Update the text of the research post.
5. Attach a backtest or notebook.
6. Click Update .
We will review your submission. If your research follows our content guidelines and provides value to the community, we may
publish it to the Research page.
Content Guidelines
To get your research onto the Research page, it must respect by the following guidelines:
1. The research is based on some financial concept, theory, or model. Itʼs not just a strategy of some technical indicators and
overfit parameters.
2. The attached backtest or notebook is concise and contains plenty of comments.
3. If the code is Python, it follows the PEP8 style guide .
4. The text is well-written English without grammar or spelling errors.
5. The text can't be generated by an LLM like ChatGPT.
6. In-line code snippets (not code blocks) are in bold face.
7. If there are math symbols throughout the text, it uses LateX syntax (for example, \(x\) ).
8. Asset class names are capitalized (for example, “Equityˮ).
9. The content has the following "Heading 1" sections:
1. Introduction : This section summarizes your area of research. Explain what type of strategy it is, the universe and
asset classes itʼs applied to, the datasets it requires, and summarize results you found. If the strategy is based on a
research paper, reference the paper at the end of the introduction.
2. Background : This section provides background information on foundational concepts of the strategy, utilizing LateX
syntax when necessary. For example, if the strategy creates low beta portfolios, this section should define what beta
is and how itʼs calculated. By the end of the Background section, readers should understand what the entire strategy
is, the universe itʼs applied to, the factors it uses, and the portfolio construction technique.
3. Implementation : This section walks the reader through how to implement the strategy in LEAN. Explain each step in
text and then include a short code snippet. When appropriate, link the content to relevant pages of the QuantConnect
documentation so readers can learn more.
4. Results : This section describes the backtest period, notes the Sharpe ratio, and explains if the strategy
underperformed or outperformed the underlying benchmark. One of the paragraphs should analyze the parameter
sensitivity and feature a screenshot of the Sharpe ratio heatmap from the optimization results page . End this section
by discussing the results and some areas of further research.
5. (Optional) References : This section is a list of references in APA style to any source material like research papers.
For more information on the APA style of reference lists, see Basic Principles of Reference List Entries on the APA
Style website.
Examples
Review the following research posts for examples that respect the content guidelines:
Low Beta Portfolios Across Industries
Opening Range Breakout for Stocks in Play
Kelly Criterion Applications in Trading Systems
Bitcoin as a Leading Indicator
Automating the Wheel Strategy
API Reference
API Reference
The QuantConnect REST API lets you communicate with our cloud servers through URL endpoints.
Authentication
Login to use the API
Project Management
Create, read, update, and delete projects
File Management
Create, read, update, and delete files from a project
Compiling Code
Compile codes for backtest and live trading
Backtest Management
Create, read, update, and delete backtests from a project
Live Management
Create, read, and update live algorithms
Optimization Management
Download data for backtest usage
Object Store Management
Generate reports for backtests to evaluate performance
Reports
Account
Lean Version
Examples
API Reference > Authentication
API Reference
Authentication
Introduction
You can make authenticated REST requests to the QuantConnect API with your User Id and API Token. You can use the
authentication endpoint described in this page to verify it is working correctly.
The base URL of QuantConnect API is https://www.quantconnect.com/api/v2 .
Follow these steps to request an API token:
1. Log in to your account.
2. In the top navigation bar, click yourUsername > My Account .
3. On your Account page , in the Security section, click Request Email With Token and Your User-Id for API Requests .
4. Click OK .
We email you your user Id and API token.
To get the organization Id, open Organization > Home and check the URL. For example, the organization Id of
https://www.quantconnect.com/organization/5cad178b20a1d52567b534553413b691 is 5cad178b20a1d52567b534553413b691.
Authenticating Requests
Requests to QuantConnect API v2 require a hashed combination of time, and the API token. The unixtime stamp combination
serves as a nonce token as each request is sent with a different signature but never requires sending the API token itself.
Hashing
Follow the below example to create a hashed token for authentication.
Make API Request
# Generate a timestamped SHA-256 hashed API token for secure authentication
from base64 import b64encode
from hashlib import sha256
from time import time
# Request your API token on https://www.quantconnect.com/settings/ and replace the below values.
USER_ID = 0
API_TOKEN = '_____'
def get_headers():
# Get timestamp
timestamp = f'{int(time())}'
time_stamped_token = f'{API_TOKEN}:{timestamp}'.encode('utf-8')
# Get hased API token
hashed_token = sha256(time_stamped_token).hexdigest()
authentication = f'{USER_ID}:{hashed_token}'.encode('utf-8')
authentication = b64encode(authentication).decode('ascii')
# Create headers dictionary.
return {
'Authorization': f'Basic {authentication}',
'Timestamp': timestamp
}
PY
Follow the below example to install the hashing into the headings and make an API request.
Authenticated State Request
Authentication status check endpoint to verify the hashing function is working successfully. The /authenticate API does not
require any information, but just an authenticated hash in the header.
Authenticated State Responses
The /authenticate API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
Example
{
"success": true
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
# Create POST Request with headers (optional: Json Content as data argument).
from requests import post
BASE_URL = 'https://www.quantconnect.com/api/v2/'
response = post(f'{BASE_URL}/<request_url>',
headers = get_headers(),
data = {},
json = {}) # Some request requires json param (must remove the data param in this case)
content = response.text
PY
API Reference > Project Management
API Reference
Project Management
The QuantConnect REST API lets you manage your projects on our cloud servers through URL endpoints.
Create Project
Read Project
Update Project
Delete Project
Collaboration
Nodes
API Reference > Project Management > Create Project
Project Management
Create Project
Introduction
Create a new project in your default organization.
Description
Create a project with the specified name and programming language. If the project-name already exists, API call returns
success:false with exception details in the errors array.
Request
Name and language of the project to create. The /projects/create API accepts requests in the following format:
CreateProjectRequest Model - Request to create a project with the specified name and language via QuantConnect.com
API.
name
string
Project name.
language
string Enum
Programming langage to use. Options : ['C#', 'Py']
organizationId
string
Optional parameter for specifying organization to create
project under. If none provided web defaults to preferred.
Example
{
"name": "string",
"language": "C#",
"organizationId": "string"
}
Responses
The /projects/create API provides a response in the following format:
200 Success
ProjectListResponse Model - Project list response.
projects
Project Array
List of projects for the authenticated user.
versions
LeanVersion Array
List of LEAN versions.
success boolean
Indicate if the API request was successful.
e
r
r
o
r
s
s
t
r
i
n
g
A
r
r
a
y
Lis
t
o
f
e
r
r
o
r
s
wit
h
t
h
e
A
PI c
all.
E
x
a
m
ple
{
"
p
r
o
j
e
c
t
s
": [
{
"
p
r
o
j
e
c
t
I
d
": 0
,
"
o
r
g
a
n
i
z
a
t
i
o
n
I
d
": 0
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
m
o
d
i
f
i
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
o
w
n
e
r
I
d
": 0
,
"
l
a
n
g
u
a
g
e
": "
C
#
"
,
"
c
o
l
l
a
b
o
r
a
t
o
r
s
": [
{
"
u
i
d
": 0
,
"
l
i
v
e
C
o
n
t
r
o
l
": t
r
u
e
,
"
p
e
r
m
i
s
s
i
o
n
": "
r
e
a
d
"
,
"
p
u
b
l
i
c
I
d
": "
s
t
r
i
n
g
"
,
"
p
r
o
f
i
l
e
I
m
a
g
e
": "https://cdn.quantconnect.c
o
m
/
w
e
b
/
i
/
u
s
e
r
s
/
p
r
o
f
i
l
e
/
a
b
c
1
2
3.j
p
e
g
"
,
"
e
m
a
i
l
": "
a
b
c
@
1
2
3.c
o
m
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
b
i
o
": "
s
t
r
i
n
g
"
,
"
o
w
n
e
r
": t
r
u
e
}
]
,
"
l
e
a
n
V
e
r
s
i
o
n
I
d
": 0
,
"
l
e
a
n
P
i
n
n
e
d
T
o
M
a
s
t
e
r
": t
r
u
e
,
"
o
w
n
e
r
": t
r
u
e
,
"
d
e
s
c
r
i
p
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
c
h
a
n
n
e
l
I
d
": "
s
t
r
i
n
g
"
,
"
p
a
r
a
m
e
t
e
r
s
": [
{
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
v
a
l
u
e
": 0
}
]
,
"
l
i
b
r
a
r
i
e
s
": [
{
"
p
r
o
j
e
c
t
I
d
": 0
,
"
l
i
b
r
a
r
y
N
a
m
e
": "
s
t
r
i
n
g
"
,
"
o
w
n
e
r
N
a
m
e
": "
s
t
r
i
n
g
"
,
"
a
c
c
e
s
s
": t
r
u
e
}
]
,
"
g
r
i
d
": "
s
t
r
i
n
g
"
,
"
l
i
v
e
G
r
i
d
": "
s
t
r
i
n
g
"
,
"
p
a
p
e
r
E
q
u
i
t
y
": 0
,
"
l
a
s
t
L
i
v
e
D
e
p
l
o
y
m
e
n
t
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
i
v
e
F
o
r
m
": ,
"
e
n
c
r
y
p
t
e
d
": t
r
u
e
,
"
c
o
d
e
R
u
n
n
i
n
g
": t
r
u
e
,
"
l
e
a
n
E
n
v
i
r
o
n
m
e
n
t
": 0
,
"
e
n
c
r
y
p
t
i
o
n
K
e
y
": {
"
i
d
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
}
}
]
,
"
v
e
r
s
i
o
n
s
": [
{
"
i
d
": ,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
d
e
s
c
r
i
p
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
l
e
a
n
H
a
s
h
": "
s
t
r
i
n
g
"
,
"
l
e
a
n
C
l
o
u
d
H
a
s
h
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
r
e
f
": "
s
t
r
i
n
g
"
,
"
p
u
b
l
i
c
": }
]
,
"
s
u
c
c
e
s
s
": t
r
u
e
,
"
e
r
r
o
r
s
": [
"
s
t
r
i
n
g
"
]
}
Project Model - Response from reading a project by id.
projectId
integer
Project id.
organizationId
integer
Orgainization id.
name
string
Name of the project.
modified
string($date-time)
Modified date for the project.
created
string($date-time)
Date the project was created.
ownerId
integer
Owner id.
language
string Enum
Programming language of the project. Options : ['C#', 'Py']
collaborators
Collaborator Array
List of collaborators.
leanVersionId
integer
The version of LEAN this project is running on.
leanPinnedToMaster
boolean
Indicate if the project is pinned to the master branch of
LEAN.
owner
boolean
Indicate if you are the owner of the project.
description
string
The project description.
channelId
string
Channel id.
parameters
ParameterSet Array
Optimization parameters.
libraries
Library Array
The library projects.
grid
string
Configuration of the backtest view grid.
liveGrid
string
Configuration of the live view grid.
paperEquity
number
The equity value of the last paper trading instance.
lastLiveDeployment string($date-time)
The last live deployment active time.
liveForm
object
The last live wizard content used.
encrypted
boolean
Indicates if the project is encrypted.
codeRunning
boolean
Indicates if the project is running or not.
leanEnvironment
integer
LEAN environment of the project running on.
encryptionKey
EncryptionKey object
Encryption key details.
Example
{
"projectId": 0,
"organizationId": 0,
"name": "string",
"modified": "2021-11-26T15:18:27.693Z",
"created": "2021-11-26T15:18:27.693Z",
"ownerId": 0,
"language": "C#",
"collaborators": [
{
"uid": 0,
"liveControl": true,
"permission": "read",
"publicId": "string",
"profileImage":
"https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg",
"email": "abc@123.com",
"name": "string",
"bio": "string",
"owner": true
}
],
"leanVersionId": 0,
"leanPinnedToMaster": true,
"owner": true,
"description": "string",
"channelId": "string",
"parameters": [
{
"name": "string",
"value": 0
}
],
"libraries": [
{
"projectId": 0,
"libraryName": "string",
"ownerName": "string",
"access": true
}
],
"grid": "string",
"liveGrid": "string",
"paperEquity": 0,
"lastLiveDeployment": "2021-11-
26T15:18:27.693Z",
"liveForm": ,
"encrypted": true,
"codeRunning": true,
"leanEnvironment": 0,
"encryptionKey": {
"id": "string",
"name": "string"
}
}
Collaborator Model
uid
integer
User ID.
liveControl
boolean
Indicate if the user have live control.
permission
string Enum
The permission this user is given. Options : ['read', 'write']
publicId
string
The user public ID.
profileImage
string
example:
https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg
The url of the user profile image.
email
string
example: abc@123.com
The registered email of the user.
name
string
The display name of the user.
bio
string
The biography of the user.
owner
boolean
Indicate if the user is the owner of the project.
Example
{
"uid": 0,
"liveControl": true,
"permission": "read",
"publicId": "string",
"profileImage":
"https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg",
"email": "abc@123.com",
"name": "string",
"bio": "string",
"owner": true
}
ParameterSet Model - Parameter set.
name
string
Name of parameter.
value
number
Value of parameter.
Example
{
"name": "string",
"value": 0
}
Library Model
projectId
integer
Project Id of the library project.
libraryName
string
Name of the library project.
ownerName
string
Name of the library project owner.
access
boolean
Indicate if the library project can be accessed.
Example
{
"projectId": 0,
"libraryName": "string",
"ownerName": "string",
"access": true
}
EncryptionKey Model - Encryption key details.
id
string
Encryption key id.
name
string
Name of the encryption key.
Example
{
"id": "string",
"name": "string"
}
LeanVersion Model
id
int
ID of the LEAN version.
created
string($date-time)
Date when this version was created.
description
string
Description of the LEAN version.
leanHash
string
Commit Hash in the LEAN repository.
leanCloudHash
string
Commit Hash in the LEAN Cloud repository.
name
string
Name of the branch where the commit is.
ref
string
Reference to the branch where the commit is.
public
int
Indicates if the version is available for the public (1) or not
(0).
Example
{
"id": ,
"created": "2021-11-26T15:18:27.693Z",
"description": "string",
"leanHash": "string",
"leanCloudHash": "string",
"name": "string",
"ref": "string",
"public":
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Project Management > Read Project
Project Management
Read Project
Introduction
List details of all projects, or the project matched the passed project ID if the project ID property passed.
Request
The projectId for the project to read, or nothing to get a details list of all projects. The /projects/read API accepts requests in
the following format:
ReadProjectRequest Model - Request to get details about a specific project or a details list of all projects.
projectId
integer
Id of the project. If not provided the API will return a details
list of all projects.
Example
{
"projectId": 0
}
Responses
The /projects/read API provides a response in the following format:
200 Success
ProjectListResponse Model - Project list response.
projects
Project Array
List of projects for the authenticated user.
versions
LeanVersion Array
List of LEAN versions.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
E
x
a
m
ple
{
"
p
r
o
j
e
c
t
s
": [
{
"
p
r
o
j
e
c
t
I
d
": 0
,
"
o
r
g
a
n
i
z
a
t
i
o
n
I
d
": 0
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
m
o
d
i
f
i
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
o
w
n
e
r
I
d
": 0
,
"
l
a
n
g
u
a
g
e
": "
C
#
"
,
"
c
o
l
l
a
b
o
r
a
t
o
r
s
": [
{
"
u
i
d
": 0
,
"
l
i
v
e
C
o
n
t
r
o
l
": t
r
u
e
,
"
p
e
r
m
i
s
s
i
o
n
": "
r
e
a
d
"
,
"
p
u
b
l
i
c
I
d
": "
s
t
r
i
n
g
"
,
"
p
r
o
f
i
l
e
I
m
a
g
e
": "https://cdn.quantconnect.c
o
m
/
w
e
b
/
i
/
u
s
e
r
s
/
p
r
o
f
i
l
e
/
a
b
c
1
2
3.j
p
e
g
"
,
"
e
m
a
i
l
": "
a
b
c
@
1
2
3.c
o
m
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
b
i
o
": "
s
t
r
i
n
g
"
,
"
o
w
n
e
r
": t
r
u
e
}
]
,
"
l
e
a
n
V
e
r
s
i
o
n
I
d
": 0
,
"
l
e
a
n
P
i
n
n
e
d
T
o
M
a
s
t
e
r
": t
r
u
e
,
"
o
w
n
e
r
": t
r
u
e
,
"
d
e
s
c
r
i
p
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
c
h
a
n
n
e
l
I
d
": "
s
t
r
i
n
g
"
,
"
p
a
r
a
m
e
t
e
r
s
": [
{
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
v
a
l
u
e
": 0
}
]
,
"
l
i
b
r
a
r
i
e
s
": [
{
"
p
r
o
j
e
c
t
I
d
": 0
,
"
l
i
b
r
a
r
y
N
a
m
e
": "
s
t
r
i
n
g
"
,
"
o
w
n
e
r
N
a
m
e
": "
s
t
r
i
n
g
"
,
"
a
c
c
e
s
s
": t
r
u
e
}
]
,
"
g
r
i
d
": "
s
t
r
i
n
g
"
,
"
l
i
v
e
G
r
i
d
": "
s
t
r
i
n
g
"
,
"
p
a
p
e
r
E
q
u
i
t
y
": 0
,
"
l
a
s
t
L
i
v
e
D
e
p
l
o
y
m
e
n
t
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
i
v
e
F
o
r
m
": ,
"
e
n
c
r
y
p
t
e
d
": t
r
u
e
,
"
c
o
d
e
R
u
n
n
i
n
g
": t
r
u
e
,
"
l
e
a
n
E
n
v
i
r
o
n
m
e
n
t
": 0
,
"
e
n
c
r
y
p
t
i
o
n
K
e
y
": {
"
i
d
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
}
}
]
,
"
v
e
r
s
i
o
n
s
": [
{
"
i
d
": ,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
d
e
s
c
r
i
p
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
l
e
a
n
H
a
s
h
": "
s
t
r
i
n
g
"
,
"
l
e
a
n
C
l
o
u
d
H
a
s
h
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
r
e
f
": "
s
t
r
i
n
g
"
,
"
p
u
b
l
i
c
": }
]
,
"
s
u
c
c
e
s
s
": t
r
u
e
,
"
e
r
r
o
r
s
": [
"
s
t
r
i
n
g
"
]
}
Project Model - Response from reading a project by id.
projectId
integer
Project id.
organizationId
integer
Orgainization id.
name
string
Name of the project.
modified
string($date-time)
Modified date for the project.
created
string($date-time)
Date the project was created.
ownerId
integer
Owner id.
language
string Enum
Programming language of the project. Options : ['C#', 'Py']
collaborators
Collaborator Array
List of collaborators.
leanVersionId
integer
The version of LEAN this project is running on.
leanPinnedToMaster
boolean
Indicate if the project is pinned to the master branch of
LEAN.
owner
boolean
Indicate if you are the owner of the project.
description
string
The project description.
channelId
string
Channel id.
parameters
ParameterSet Array
Optimization parameters.
libraries
Library Array
The library projects.
grid
string
Configuration of the backtest view grid.
liveGrid
string
Configuration of the live view grid.
paperEquity
number
The equity value of the last paper trading instance.
lastLiveDeployment
string($date-time)
The last live deployment active time.
liveForm
object
The last live wizard content used.
encrypted
boolean
Indicates if the project is encrypted.
codeRunning
boolean
Indicates if the project is running or not.
leanEnvironment
integer
LEAN environment of the project running on.
encryptionKey
EncryptionKey object
Encryption key details.
E
x
a
m
ple
{
"
p
r
o
j
e
c
t
I
d
": 0
,
"
o
r
g
a
n
i
z
a
t
i
o
n
I
d
": 0
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
m
o
d
i
f
i
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
o
w
n
e
r
I
d
": 0
,
"
l
a
n
g
u
a
g
e
": "
C
#
"
,
"
c
o
l
l
a
b
o
r
a
t
o
r
s
": [
{
"
u
i
d
": 0
,
"
l
i
v
e
C
o
n
t
r
o
l
": t
r
u
e
,
"
p
e
r
m
i
s
s
i
o
n
": "
r
e
a
d
"
,
"
p
u
b
l
i
c
I
d
": "
s
t
r
i
n
g
"
,
"
p
r
o
f
i
l
e
I
m
a
g
e
": "https://cdn.quantcon
n
e
c
t.c
o
m
/
w
e
b
/
i
/
u
s
e
r
s
/
p
r
o
f
i
l
e
/
a
b
c
1
2
3.j
p
e
g
"
,
"
e
m
a
i
l
": "
a
b
c
@
1
2
3.c
o
m
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
b
i
o
": "
s
t
r
i
n
g
"
,
"
o
w
n
e
r
": t
r
u
e
}
]
,
"
l
e
a
n
V
e
r
s
i
o
n
I
d
": 0
,
"
l
e
a
n
P
i
n
n
e
d
T
o
M
a
s
t
e
r
": t
r
u
e
,
"
o
w
n
e
r
": t
r
u
e
,
"
d
e
s
c
r
i
p
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
c
h
a
n
n
e
l
I
d
": "
s
t
r
i
n
g
"
,
"
p
a
r
a
m
e
t
e
r
s
": [
{
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
v
a
l
u
e
": 0
}
]
,
"
l
i
b
r
a
r
i
e
s
": [
{
"
p
r
o
j
e
c
t
I
d
": 0
,
"
l
i
b
r
a
r
y
N
a
m
e
": "
s
t
r
i
n
g
"
,
"
o
w
n
e
r
N
a
m
e
": "
s
t
r
i
n
g
"
,
"
a
c
c
e
s
s
": t
r
u
e
}
]
,
"
g
r
i
d
": "
s
t
r
i
n
g
"
,
"
l
i
v
e
G
r
i
d
": "
s
t
r
i
n
g
"
,
"
p
a
p
e
r
E
q
u
i
t
y
": 0
,
"
l
a
s
t
L
i
v
e
D
e
p
l
o
y
m
e
n
t
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
i
v
e
F
o
r
m
": ,
"
e
n
c
r
y
p
t
e
d
": t
r
u
e
,
"
c
o
d
e
R
u
n
n
i
n
g
": t
r
u
e
,
"
l
e
a
n
E
n
v
i
r
o
n
m
e
n
t
": 0
,
"
e
n
c
r
y
p
t
i
o
n
K
e
y
": {
"
i
d
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
}
}
Collaborator Model
uid
integer
User ID.
liveControl
boolean
Indicate if the user have live control.
permission
string Enum
The permission this user is given. Options : ['read', 'write']
publicId
string
The user public ID.
profileImage
string
example:
https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg
The url of the user profile image.
email
string
example: abc@123.com
The registered email of the user.
name
string
The display name of the user.
bio
string
The biography of the user.
owner
boolean
Indicate if the user is the owner of the project.
Example
{
"uid": 0,
"liveControl": true,
"permission": "read",
"publicId": "string",
"profileImage":
"https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg",
"email": "abc@123.com",
"name": "string",
"bio": "string",
"owner": true
}
ParameterSet Model - Parameter set.
name
string
Name of parameter.
value
number
Value of parameter.
Example
{
"name": "string",
"value": 0
}
Library Model
projectId
integer
Project Id of the library project.
libraryName
string
Name of the library project.
ownerName
string
Name of the library project owner.
access
boolean
Indicate if the library project can be accessed.
Example
{
"projectId": 0,
"libraryName": "string",
"ownerName": "string",
"access": true
}
EncryptionKey Model - Encryption key details.
id
string
Encryption key id.
name
string
Name of the encryption key.
Example
{
"id": "string",
"name": "string"
}
LeanVersion Model
id
int
ID of the LEAN version.
created
string($date-time)
Date when this version was created.
description
string
Description of the LEAN version.
leanHash
string
Commit Hash in the LEAN repository.
leanCloudHash
string
Commit Hash in the LEAN Cloud repository.
name
string
Name of the branch where the commit is.
ref
string
Reference to the branch where the commit is.
public
int
Indicates if the version is available for the public (1) or not
(0).
Example
{
"id": ,
"created": "2021-11-26T15:18:27.693Z",
"description": "string",
"leanHash": "string",
"leanCloudHash": "string",
"name": "string",
"ref": "string",
"public":
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Project Management > Update Project
Project Management
Update Project
Introduction
Update a project name, description or parameters.
Request
The /projects/update API accepts requests in the following format:
UpdateProjectRequest Model - Update a project name, description or parameters.
projectId
integer
Project Id to which the file belongs.
name
string
The new name for the project.
description
object
The new description for the project.
Example
{
"projectId": 0,
"name": "string",
"description":
}
Responses
The /projects/update API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Project Management > Delete Project
Project Management
Delete Project
Introduction
Delete a project.
Request
The /projects/delete API accepts requests in the following format:
DeleteProjectRequest Model - Request to delete a project.
projectId
integer
Project Id to which the file belongs.
Example
{
"projectId": 0
}
Responses
The /projects/delete API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
Example
This example demonstrates how to delete projects with a given word in its name using the QuantConnect API.
from base64 import b64encode
from hashlib import sha256
from time import time
from requests import get, post
BASE_URL = 'https://www.quantconnect.com/api/v2/'
# You need to replace these with your actual credentials.
# You can request your credentials at https://www.quantconnect.com/settings/
# You can find our organization ID at https://www.quantconnect.com/organization/
USER_ID = 0
API_TOKEN = '____'
ORGANIZATION_ID = '____'
# Define a key to filter projects by name.
# This key will be used to search for projects that contain this string in their name.
PROJECT_NAME_SEARCH_KEY = 'unsorted'
def get_headers():
# Get timestamp
timestamp = f'{int(time())}'
time_stamped_token = f'{API_TOKEN}:{timestamp}'.encode('utf-8')
# Get hased API token
hashed_token = sha256(time_stamped_token).hexdigest()
authentication = f'{USER_ID}:{hashed_token}'.encode('utf-8')
authentication = b64encode(authentication).decode('ascii')
# Create headers dictionary.
return {
'Authorization': f'Basic {authentication}',
'Timestamp': timestamp
}
# Authenticate
response = post(f'{BASE_URL}/authenticate', headers = get_headers())
print(response.json())
def delete_project(project_id):
"""
Deletes a project by its ID.
:param project_id: The ID of the project to delete.
:return: Response from the API.
"""
response = post(f'{BASE_URL}/projects/delete', headers=get_headers(), data={'projectId': project_id})
return response.json()
def list_projects(organization_id):
"""
Lists all projects for the authenticated user.
:return: Response from the API containing the list of projects.
"""
response = get(f'{BASE_URL}/projects/read', headers=get_headers())
projects = response.json().get('projects', [])
# Filter projects by organization ID to ensure we only get projects belonging to the specified organization.
return [p for p in projects if p['organizationId'] == organization_id]
if __name__ == "__main__":
# List all projects of the organization
projects = list_projects(ORGANIZATION_ID)
print("All Projects:", len(projects))
# Filter projects by a specific key in the name
projects = [p for p in projects if PROJECT_NAME_SEARCH_KEY in p['name']]
print("Filtered Projects:", len(projects))
for project in projects:
# Delete a specific project by ID
project_id_to_delete = project['projectId']
delete_response = delete_project(project_id_to_delete)
print(f"Project ID: {project_id_to_delete}, Name: {project['name']}, Delete Response: {delete_response}")
PY
API Reference > Project Management > Collaboration
Project Management
Collaboration
API Reference > Project Management > Collaboration > Create Project Collaborator
Collaboration
Create Project Collaborator
Introduction
Adds collaborator to the project.
Request
The /projects/collaboration/create API accepts requests in the following format:
CreateCollaboratorRequest Model - Request to create a new backtest.
projectId
integer
Project Id we sent for compile.
collaboratorUserId
string
User Id of the collaborator we want to add.
collaborationLiveControl
bool
Gives the right to deploy and stop live algorithms.
collaborationWrite
bool
Gives the right to edit the code.
Example
{
"projectId": 0,
"collaboratorUserId": "string",
"collaborationLiveControl": ,
"collaborationWrite":
}
Responses
The /projects/collaboration/create API provides a response in the following format:
200 Success
CreateCollaboratorResponse Model - Response received when creating collaborator.
collaborators
Collaborator Array
List of collaborators.
success
boolean
Indicate if the API request was successful.
Example
{
"collaborators": [
{
"uid": 0,
"liveControl": true,
"permission": "read",
"publicId": "string",
"profileImage":
"https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg",
"email": "abc@123.com",
"name": "string",
"bio": "string",
"owner": true
}
],
"success": true
}
Collaborator Model
uid
integer
User ID.
liveControl
boolean
Indicate if the user have live control.
permission
string Enum
The permission this user is given. Options : ['read', 'write']
publicId
string
The user public ID.
profileImage
string
example:
https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg
The url of the user profile image.
email
string
example: abc@123.com
The registered email of the user.
name
string
The display name of the user.
bio
string
The biography of the user.
owner
boolean
Indicate if the user is the owner of the project.
Example
{
"uid": 0,
"liveControl": true,
"permission": "read",
"publicId": "string",
"profileImage":
"https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg",
"email": "abc@123.com",
"name": "string",
"bio": "string",
"owner": true
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Project Management > Collaboration > Read Project Collaborators
Collaboration
Read Project Collaborators
Introduction
List all collaborators in a project.
Request
The /projects/collaboration/read API accepts requests in the following format:
ReadCollaboratorsRequest Model - Request to list the collaborators in a project.
projectId
integer
Id of the project from which to read one or multiple
collaborators.
Example
{
"projectId": 0
}
Responses
The /projects/collaboration/read API provides a response in the following format:
200 Success
ReadCollaboratorsResponse Model - Response received when reading the collaborators of a project.
collaborators
Collaborator Array
List of collaborators.
userLiveControl
boolean
Indicate if the user has the right to deploy and stop live
algorithms.
userPermissions
string
List the user permissions - write/read.
success
boolean
Indicate if the API request was successful.
Example
{
"collaborators": [
{
"uid": 0,
"liveControl": true,
"permission": "read",
"publicId": "string",
"profileImage":
"https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg",
"email": "abc@123.com",
"name": "string",
"bio": "string",
"owner": true
}
],
"userLiveControl": true,
"userPermissions": "string",
"success": true
}
Collaborator Model
uid
integer
User ID.
liveControl
boolean
Indicate if the user have live control.
permission
string Enum
The permission this user is given. Options : ['read', 'write']
publicId
string
The user public ID.
profileImage
string
example:
https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg
The url of the user profile image.
email
string
example: abc@123.com
The registered email of the user.
name
string
The display name of the user.
bio
string
The biography of the user.
owner
boolean
Indicate if the user is the owner of the project.
Example
{
"uid": 0,
"liveControl": true,
"permission": "read",
"publicId": "string",
"profileImage":
"https://cdn.quantconnect.com/web/i/users/profile/abc123.jpeg",
"email": "abc@123.com",
"name": "string",
"bio": "string",
"owner": true
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Project Management > Nodes
Project Management
Nodes
API Reference > Project Management > Nodes > Read Project Nodes
Nodes
Read Project Nodes
Introduction
Read all nodes in a project.
Request
The /projects/nodes/read API accepts requests in the following format:
ReadProjectNodesRequest Model - Request to get details about nodes of a specific organization.
projectId
string
Project Id to which the nodes refer.
Example
{
"projectId": "string"
}
Responses
The /projects/nodes/read API provides a response in the following format:
200 Success
ProjectNodesResponse Model - Response received when reading all nodes of a project.
nodes
#/components/schemas/ProjectNodes
List of project nodes.
autoSelectNode
boolean
Indicate if a node is automatically selected.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"nodes": ,
"autoSelectNode": true,
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Project Management > Nodes > Update Project Nodes
Nodes
Update Project Nodes
Introduction
Update the active state of some nodes to true. If you don't provide any nodes, all the nodes become inactive and
AutoSelectNode is true.
Request
The /projects/nodes/update API accepts requests in the following format:
UpdateProjectNodesRequest Model - Request to update the nodes of a project.
projectId
integer
Project Id to which the nodes refer.
nodes
string Array
List of node Id to update.
Example
{
"projectId": 0,
"nodes": [
"string"
]
}
Responses
The /projects/nodes/update API provides a response in the following format:
200 Success
ProjectNodesResponse Model - Response received when reading all nodes of a project.
nodes
#/components/schemas/ProjectNodes
List of project nodes.
autoSelectNode
boolean
Indicate if a node is automatically selected.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"nodes": ,
"autoSelectNode": true,
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > File Management
API Reference
File Management
The QuantConnect REST API lets you manage your files on our cloud servers through URL endpoints.
Create File
Read File
Update File
Delete File
API Reference > File Management > Create File
File Management
Create File
Introduction
Add a file to given project.
Request
Project, file name and file content to create. The /files/create API accepts requests in the following format:
CreateProjectFileRequest Model - Request to add a file to a project.
projectId
integer
Project Id to which the file belongs.
name
string
example:
main.py
The name of the new file.
content
string
The content of the new file.
Example
{
"projectId": 0,
"name": "main.py",
"content": "string"
}
Responses
The /files/create API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > File Management > Read File
File Management
Read File
Introduction
If a ReadSingleFileRequest is passed, reads that file from the project. If a ReadAllFilesRequest is passed, reads all files in the
project.
Request
An array list of files from the project requested. The /files/read API accepts requests in the following format:
ReadFilesRequest Model - Request to read all files from a project or just one (if the name is provided).
projectId
integer
Project Id to which the file belongs.
name
string
Optional. The name of the file that will be read.
Example
{
"projectId": 0,
"name": "string"
}
Responses
The /files/read API provides a response in the following format:
200 Success
ProjectFilesResponse Model - Response received when reading files from a project.
files
ProjectFile Array
List of project file information.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"files": [
{
"id": ,
"projectId": ,
"name": "string",
"content": "string",
"modified": "2021-11-26T15:18:27.693Z",
"open": true,
"isLibrary": true
}
],
"success": true,
"errors": [
"string"
]
}
ProjectFile Model - File for a project.
id
int
ID of the project file. This can also be null.
projectId
int
ID of the project.
name
string
Name of a project file.
content
string
Contents of the project file.
modified
string($date-time)
DateTime project file was modified.
open
boolean
Indicates if the project file is open or not.
isLibrary
boolean
Indicates if the project file is a library or not. It's always
false in live/read and backtest/read.
Example
{
"id": ,
"projectId": ,
"name": "string",
"content": "string",
"modified": "2021-11-26T15:18:27.693Z",
"open": true,
"isLibrary": true
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > File Management > Update File
File Management
Update File
Introduction
If an UpdateProjectFileNameRequest is passed, update the name of a file. If a UpdateProjectFileContentsRequest is passed,
update the contents of a file.
Request
Information about the file to update along with the new properties to set. The /files/update API accepts requests in the
following format:
UpdateFileNameRequest Model - Request to update the name of a file.
projectId
integer
Project Id to which the file belongs.
oldFileName
string
The current name of the file.
newName
string
The new name for the file.
Example
{
"projectId": 0,
"oldFileName": "string",
"newName": "string"
}
UpdateFileContentsRequest Model - Request to update the contents of a file.
projectId
integer
Project Id to which the file belongs.
name
string
The name of the file that should be updated.
content
string
The new contents of the file.
Example
{
"projectId": 0,
"name": "string",
"content": "string"
}
Responses
The /files/update API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > File Management > Delete File
File Management
Delete File
Introduction
Delete a file in a project.
Request
Project Id and filename to specify the file for deletion. The /files/delete API accepts requests in the following format:
DeleteFileRequest Model - Request to delete a file in a project.
projectId
integer
Project Id to which the file belongs.
name
string
The name of the file that should be deleted.
Example
{
"projectId": 0,
"name": "string"
}
Responses
The /files/delete API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Compiling Code
API Reference
Compiling Code
The QuantConnect REST API lets you compile your projects on our cloud servers through URL endpoints.
Create Compilation Job
Read Compilation Result
API Reference > Compiling Code > Create Compilation Job
Compiling Code
Create Compilation Job
Introduction
Asynchronously create a compile job request for a project.
Request
Project Id specifying project to build. The /compile/create API accepts requests in the following format:
CreateCompileRequest Model - Request to compile a project.
projectId
integer
Project Id we wish to compile.
Example
{
"projectId": 0
}
Responses
The /compile/create API provides a response in the following format:
200 Success
CompileResponse Model - Response from the compiler on a build event.
compileId
string
Compile Id for a sucessful build.
state
string Enum
True on successful compile. Options : ['InQueue',
'BuildSuccess', 'BuildError']
projectId
integer
Project Id we sent for compile.
signature
string
Signature key of compilation.
signatureOrder
string Array
Signature order of files to be compiled.
logs
string Array
Logs of the compilation request.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"compileId": "string",
"state": "InQueue",
"projectId": 0,
"signature": "string",
"signatureOrder": [
"string"
],
"logs": [
"string"
],
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Compiling Code > Read Compilation Result
Compiling Code
Read Compilation Result
Introduction
Read a compile packet job result.
Request
Read a compile result for a specific Project Id and Compile Id. The /compile/read API accepts requests in the following format:
ReadCompileRequest Model - Request to read a compile packet job.
projectId
integer
Project Id we sent for compile.
compileId
string
Compile Id returned during the creation request.
Example
{
"projectId": 0,
"compileId": "string"
}
Responses
The /compile/read API provides a response in the following format:
200 Success
CompileResponse Model - Response from the compiler on a build event.
compileId
string
Compile Id for a sucessful build.
state
string Enum
True on successful compile. Options : ['InQueue',
'BuildSuccess', 'BuildError']
projectId
integer
Project Id we sent for compile.
signature
string
Signature key of compilation.
signatureOrder
string Array
Signature order of files to be compiled.
logs
string Array
Logs of the compilation request.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"compileId": "string",
"state": "InQueue",
"projectId": 0,
"signature": "string",
"signatureOrder": [
"string"
],
"logs": [
"string"
],
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management
API Reference
Backtest Management
The QuantConnect REST API lets you manage backtests on our cloud servers through URL endpoints.
Create Backtest
Read Backtest
Update Backtest
Delete Backtest
List Backtests
Introduction
Updates the tags collection for a backtest.
Request
Information required to update the tags collection for a backtest. The /backtests/tags/update API accepts requests in the
following format:
UpdateBacktestTagsRequest Model - Updates the tags collection for a backtest.
projectId
integer
Project Id for the backtest we want to update.
backtestId
Backtest Id we want to update.
/.
tags
string Array
Array of the new backtest tags.
Example
{
"projectId": 0,
"backtestId": ,
"tags": [
"string"
]
}
Responses
The /backtests/tags/update API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > Create Backtest
Backtest Management
Create Backtest
Introduction
Create a new backtest request and get the backtest Id.
Request
Create a new backtest given a project Id and compile Id. The /backtests/create API accepts requests in the following format:
CreateBacktestRequest Model - Request to create a new backtest.
projectId
integer
Project Id we sent for compile.
compileId
string
Compile Id for the project to backtest.
backtestName
string
Name for the new backtest.
parameters[name]
object
example: value
Parameters used in the backtest. E.g.,
parameters[ema_fast] = 10, parameters[ema_slow] = 100,
etc.
Example
{
"projectId": 0,
"compileId": "string",
"backtestName": "string",
"parameters[name]": "value"
}
Responses
The /backtests/create API provides a response in the following format:
200 Success
BacktestResponse Model - Collection container for a list of backtests for a project.
backtest
BacktestResult Array
Collection of backtests for a project.
debugging
boolean
Indicates if the backtest is run under debugging mode.
success
boolean
Indicate if the API request was successful.
e
r
r
o
r
s
s
t
r
i
n
g
A
r
r
a
y
Lis
t
o
f
e
r
r
o
r
s
wit
h
t
h
e
A
PI c
all. { "backtest": [ { "note": "string", "name": "string", "organizationId": 0
,
"
p
r
o
j
e
c
t
I
d
": 0
,
"
c
o
m
p
l
e
t
e
d
": t
r
u
e
,
"
o
p
t
i
m
i
z
a
t
i
o
n
I
d
": "
s
t
r
i
n
g
"
,
"
b
a
c
k
t
e
s
t
I
d
": "
s
t
r
i
n
g
"
,
"
t
r
a
d
e
a
b
l
e
D
a
t
e
s
": 0
,
"
r
e
s
e
a
r
c
h
G
u
i
d
e
": {
"
m
i
n
u
t
e
s
": 0
,
"
b
a
c
k
t
e
s
t
C
o
u
n
t
": 0
,
"
p
a
r
a
m
e
t
e
r
s
": 0
}
,
"
b
a
c
k
t
e
s
t
S
t
a
r
t
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
b
a
c
k
t
e
s
t
E
n
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
s
n
a
p
s
h
o
t
I
d
": 0
,
"
s
t
a
t
u
s
": "
C
o
m
p
l
e
t
e
d."
,
"
e
r
r
o
r
": "
s
t
r
i
n
g
"
,
"
s
t
a
c
k
t
r
a
c
e
": "
s
t
r
i
n
g
"
,
"
p
r
o
g
r
e
s
s
": 0
,
"
h
a
s
I
n
i
t
i
a
l
i
z
e
E
r
r
o
r
": t
r
u
e
,
"
c
h
a
r
t
s
": {
"
n
a
m
e
": "
s
t
r
i
n
g
"
}
,
"
p
a
r
a
m
e
t
e
r
S
e
t
": {
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
v
a
l
u
e
": 0
}
,
"
r
o
l
l
i
n
g
W
i
n
d
o
w
": {
"
t
r
a
d
e
S
t
a
t
i
s
t
i
c
s
": {
"
s
t
a
r
t
D
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
n
d
D
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
t
o
t
a
l
N
u
m
b
e
r
O
f
T
r
a
d
e
s
": 0
,
"
n
u
m
b
e
r
O
f
W
i
n
n
i
n
g
T
r
a
d
e
s
": 0
,
"
n
u
m
b
e
r
O
f
L
o
s
i
n
g
T
r
a
d
e
s
": 0
,
"
t
o
t
a
l
P
r
o
f
i
t
L
o
s
s
": 0
,
"
t
o
t
a
l
P
r
o
f
i
t
": 0
,
"
t
o
t
a
l
L
o
s
s
": 0
,
"
l
a
r
g
e
s
t
P
r
o
f
i
t
": 0
,
"
l
a
r
g
e
s
t
L
o
s
s
": 0
,
"
a
v
e
r
a
g
e
P
r
o
f
i
t
L
o
s
s
": 0
,
"
a
v
e
r
a
g
e
P
r
o
f
i
t
": 0
,
"
a
v
e
r
a
g
e
L
o
s
s
": 0
,
"
a
v
e
r
a
g
e
T
r
a
d
e
D
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
a
v
e
r
a
g
e
W
i
n
n
i
n
g
T
r
a
d
e
D
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
a
v
e
r
a
g
e
L
o
s
i
n
g
T
r
a
d
e
D
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
e
d
i
a
n
T
r
a
d
e
D
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
e
d
i
a
n
W
i
n
n
i
n
g
T
r
a
d
e
D
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
e
d
i
a
n
L
o
s
i
n
g
T
r
a
d
e
D
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
a
x
C
o
n
s
e
c
u
t
i
v
e
W
i
n
n
i
n
g
T
r
a
d
e
s
": 0
,
"
m
a
x
C
o
n
s
e
c
u
t
i
v
e
L
o
s
i
n
g
T
r
a
d
e
s
": 0
,
"
p
r
o
f
i
t
L
o
s
s
R
a
t
i
o
": 0
,
"
w
i
n
L
o
s
s
R
a
t
i
o
": 0
,
"
w
i
n
R
a
t
e
": 0
,
"
l
o
s
s
R
a
t
e
": 0
,
"
a
v
e
r
a
g
e
M
A
E
": 0
,
"
a
v
e
r
a
g
e
M
F
E
": 0
,
"
l
a
r
g
e
s
t
M
A
E
": 0
,
"
l
a
r
g
e
s
t
M
F
E
": 0
,
"
m
a
x
i
m
u
m
C
l
o
s
e
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
,
"
m
a
x
i
m
u
m
I
n
t
r
a
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
,
"
p
r
o
f
i
t
L
o
s
s
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": 0
,
"
p
r
o
f
i
t
L
o
s
s
D
o
w
n
s
i
d
e
D
e
v
i
a
t
i
o
n
": 0
,
"
p
r
o
f
i
t
F
a
c
t
o
r
": 0
,
"
s
h
a
r
p
e
R
a
t
i
o
": 0
,
"
s
o
r
t
i
n
o
R
a
t
i
o
": 0
,
"
p
r
o
f
i
t
T
o
M
a
x
D
r
a
w
d
o
w
n
R
a
t
i
o
": 0
,
"
m
a
x
i
m
u
m
E
n
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
,
E
x
a
m
ple
"
m
a
x
i
m
u
m
E
n
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
,
"
a
v
e
r
a
g
e
E
n
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
,
"
m
a
x
i
m
u
m
D
r
a
w
d
o
w
n
D
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
t
o
t
a
l
F
e
e
s
": 0
}
,
"
p
o
r
t
f
o
l
i
o
S
t
a
t
i
s
t
i
c
s
": {
"
a
v
e
r
a
g
e
W
i
n
R
a
t
e
": 0
,
"
a
v
e
r
a
g
e
L
o
s
s
R
a
t
e
": 0
,
"
p
r
o
f
i
t
L
o
s
s
R
a
t
i
o
": 0
,
"
w
i
n
R
a
t
e
": 0
,
"
l
o
s
s
R
a
t
e
": 0
,
"
e
x
p
e
c
t
a
n
c
y
": 0
,
"
s
t
a
r
t
E
q
u
i
t
y
": 0
,
"
e
n
d
E
q
u
i
t
y
": 0
,
"
c
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": 0
,
"
d
r
a
w
d
o
w
n
": 0
,
"
t
o
t
a
l
N
e
t
P
r
o
f
i
t
": 0
,
"
s
h
a
r
p
e
R
a
t
i
o
": 0
,
"
p
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": 0
,
"
s
o
r
t
i
n
o
R
a
t
i
o
": 0
,
"
a
l
p
h
a
": 0
,
"
b
e
t
a
": 0
,
"
a
n
n
u
a
l
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": 0
,
"
a
n
n
u
a
l
V
a
r
i
a
n
c
e
": 0
,
"
i
n
f
o
r
m
a
t
i
o
n
R
a
t
i
o
": 0
,
"
t
r
a
c
k
i
n
g
E
r
r
o
r
": 0
,
"
t
r
e
y
n
o
r
R
a
t
i
o
": 0
,
"
p
o
r
t
f
o
l
i
o
T
u
r
n
o
v
e
r
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
9
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
5
": 0
}
,
"
c
l
o
s
e
d
T
r
a
d
e
s
": [
{
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
e
n
t
r
y
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
n
t
r
y
P
r
i
c
e
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
L
o
n
g
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
e
x
i
t
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
x
i
t
P
r
i
c
e
": 0
,
"
p
r
o
f
i
t
L
o
s
s
": 0
,
"
t
o
t
a
l
F
e
e
s
": 0
,
"
m
a
e
": 0
,
"
m
f
e
": 0
,
"
d
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
e
n
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
}
]
}
,
"
r
u
n
t
i
m
e
S
t
a
t
i
s
t
i
c
s
": {
"
E
q
u
i
t
y
": "
$
1
0
0.0
0
"
,
"
F
e
e
s
": "
-
$
1
0
0.0
0
"
,
"
H
o
l
d
i
n
g
s
": "
$
1
0
0.0
0
"
,
"
N
e
t
P
r
o
f
i
t
": "
$
1
0
0.0
0
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
5
0.0
0
%
"
,
"
R
e
t
u
r
n
": "
5
0.0
0
%
"
,
"
U
n
r
e
a
l
i
z
e
d
": "
$
1
0
0.0
0
"
,
"
V
o
l
u
m
e
": "
$
1
0
0.0
0
"
}
,
"
s
t
a
t
i
s
t
i
c
s
": {
"
T
o
t
a
l
O
r
d
e
r
s
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
W
i
n
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
L
o
s
s
": "
s
t
r
i
n
g
"
,
"
C
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": "
s
t
r
i
n
g
"
,
"
D
r
a
w
d
o
w
n
": "
s
t
r
i
n
g
"
,
"
E
x
p
e
c
t
a
n
c
y
": "
s
t
r
i
n
g
"
,
"
S
t
a
r
t
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
E
n
d
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
N
e
t
P
r
o
f
i
t
": "
s
t
r
i
n
g
"
,
"
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
S
o
r
t
i
n
o
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
L
o
s
s
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
W
i
n
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
P
r
o
f
i
t
-
L
o
s
s
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
A
l
p
h
a
": "
s
t
r
i
n
g
"
,
"Alpha": "string",
"Beta": "string",
"Annual Standard Deviation": "string",
"Annual Variance": "string",
"Information Ratio": "string",
"Tracking Error": "string",
"Treynor Ratio": "string",
"Total Fees": "string",
"Estimated Strategy Capacity": "string",
"Lowest Capacity Asset": "string",
"Portfolio Turnover": "string"
},
"totalPerformance": {
"tradeStatistics": {
"startDateTime": "2021-11-
26T15:18:27.693Z",
"endDateTime": "2021-11-
26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
},
"closedTrades": [
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-
26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-
26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
]
},
"nodeName": "string",
"outOfSampleMaxEndDate": "2021-11-
26T15:18:27.693Z",
"outOfSampleDays": 0
}
],
"debugging": true,
"success": true,
"errors": [
"string"
]
}
BacktestResult Model - Results object class. Results are exhaust from backtest or live algorithms running in LEAN.
note
string
Note on the backtest attached by the user.
name
string
Name of the backtest.
organizationId
integer
Organization ID.
projectId
integer
Project ID.
completed
boolean
Boolean true when the backtest is completed.
optimizationId
string
Optimization task ID, if the backtest is part of an
optimization.
backtestId
string
Assigned backtest ID.
tradeableDates
integer
Number of traadeable days.
researchGuide
ResearchGuide object
A power gauge for backtests, time and parameters to
estimate the overfitting risk.
backtestStart
string($date-time)
The starting time of the backtest.
backtestEnd
string($date-time)
The ending time of the backtest.
created
string($date-time)
Backtest creation date and time.
snapshotId
integer
Snapshot id of this backtest result.
status
string Enum
Status of the backtest. Options : ['Completed.', 'In
Queue...', "'Running: _%'"]
error
string
Backtest error message.
stacktrace
string
Backtest error stacktrace.
progress
number
Progress of the backtest in percent 0-1.
hasInitializeError
boolean
Indicates if the backtest has error during initialization.
charts
ChartSummary object
Charts updates for the live algorithm since the last result
packet.
parameterSet
ParameterSet object
Parameters used in the backtest.
rollingWindow
AlgorithmPerformance object
Rolling window detailed statistics.
runtimeStatistics
RuntimeStatistics object
Runtime banner/updating statistics in the title banner of the
live algorithm GUI.
statistics
StatisticsResult object
Statistics information sent during the algorithm operations.
totalPerformance
AlgorithmPerformance object
The algorithm performance statistics.
nodeName
string
The backtest node name.
outOfSampleMaxEndDate
string($date-time)
End date of out of sample data.
outOfSampleDays
integer
Number of days of out of sample days.
{
"note": "string",
"name": "string",
"organizationId": 0,
"projectId": 0,
"completed": true,
"optimizationId": "string",
"backtestId": "string",
"tradeableDates": 0,
"researchGuide": {
"minutes": 0,
"backtestCount": 0,
"parameters": 0
},
"backtestStart": "2021-11-26T15:18:27.693Z",
"backtestEnd": "2021-11-26T15:18:27.693Z",
"created": "2021-11-26T15:18:27.693Z",
"snapshotId": 0,
"status": "Completed.",
"error": "string",
"stacktrace": "string",
"progress": 0,
"hasInitializeError": true,
"charts": {
"name": "string"
},
"parameterSet": {
"name": "string",
"value": 0
},
"rollingWindow": {
"tradeStatistics": {
"startDateTime": "2021-11-26T15:18:27.693Z",
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
E
x
a
m
ple
"
w
i
n
R
a
t
e
": 0
,
"
l
o
s
s
R
a
t
e
": 0
,
"
e
x
p
e
c
t
a
n
c
y
": 0
,
"
s
t
a
r
t
E
q
u
i
t
y
": 0
,
"
e
n
d
E
q
u
i
t
y
": 0
,
"
c
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": 0
,
"
d
r
a
w
d
o
w
n
": 0
,
"
t
o
t
a
l
N
e
t
P
r
o
f
i
t
": 0
,
"
s
h
a
r
p
e
R
a
t
i
o
": 0
,
"
p
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": 0
,
"
s
o
r
t
i
n
o
R
a
t
i
o
": 0
,
"
a
l
p
h
a
": 0
,
"
b
e
t
a
": 0
,
"
a
n
n
u
a
l
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": 0
,
"
a
n
n
u
a
l
V
a
r
i
a
n
c
e
": 0
,
"
i
n
f
o
r
m
a
t
i
o
n
R
a
t
i
o
": 0
,
"
t
r
a
c
k
i
n
g
E
r
r
o
r
": 0
,
"
t
r
e
y
n
o
r
R
a
t
i
o
": 0
,
"
p
o
r
t
f
o
l
i
o
T
u
r
n
o
v
e
r
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
9
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
5
": 0
}
,
"
c
l
o
s
e
d
T
r
a
d
e
s
": [
{
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
e
n
t
r
y
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
n
t
r
y
P
r
i
c
e
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
L
o
n
g
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
e
x
i
t
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
x
i
t
P
r
i
c
e
": 0
,
"
p
r
o
f
i
t
L
o
s
s
": 0
,
"
t
o
t
a
l
F
e
e
s
": 0
,
"
m
a
e
": 0
,
"
m
f
e
": 0
,
"
d
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
e
n
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
}
]
}
,
"
r
u
n
t
i
m
e
S
t
a
t
i
s
t
i
c
s
": {
"
E
q
u
i
t
y
": "
$
1
0
0.0
0
"
,
"
F
e
e
s
": "
-
$
1
0
0.0
0
"
,
"
H
o
l
d
i
n
g
s
": "
$
1
0
0.0
0
"
,
"
N
e
t
P
r
o
f
i
t
": "
$
1
0
0.0
0
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
5
0.0
0
%
"
,
"
R
e
t
u
r
n
": "
5
0.0
0
%
"
,
"
U
n
r
e
a
l
i
z
e
d
": "
$
1
0
0.0
0
"
,
"
V
o
l
u
m
e
": "
$
1
0
0.0
0
"
}
,
"
s
t
a
t
i
s
t
i
c
s
": {
"
T
o
t
a
l
O
r
d
e
r
s
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
W
i
n
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
L
o
s
s
": "
s
t
r
i
n
g
"
,
"
C
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": "
s
t
r
i
n
g
"
,
"
D
r
a
w
d
o
w
n
": "
s
t
r
i
n
g
"
,
"
E
x
p
e
c
t
a
n
c
y
": "
s
t
r
i
n
g
"
,
"
S
t
a
r
t
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
E
n
d
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
N
e
t
P
r
o
f
i
t
": "
s
t
r
i
n
g
"
,
"
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
S
o
r
t
i
n
o
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
L
o
s
s
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
W
i
n
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
P
r
o
f
i
t
-
L
o
s
s
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
A
l
p
h
a
": "
s
t
r
i
n
g
"
,
"
B
e
t
a
": "
s
t
r
i
n
g
"
,
"
A
n
n
u
a
l
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
A
n
n
u
a
l
V
a
r
i
a
n
c
e
": "
s
t
r
i
n
g
"
,
"
I
n
f
o
r
m
a
t
i
o
n
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
T
r
a
c
k
i
n
g
E
r
r
o
r
": "
s
t
r
i
n
g
"
,
"
T
r
e
y
n
o
r
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
T
o
t
a
l
F
e
e
s
": "
s
t
r
i
n
g
"
,
"
E
s
t
i
m
a
t
e
d
S
t
r
a
t
e
g
y
C
a
p
a
c
i
t
y
": "
s
t
r
i
n
g
"
,
"
L
o
w
e
s
t
C
a
p
a
c
i
t
y
A
s
s
e
t
": "
s
t
r
i
n
g
"
,
"
P
o
r
t
f
o
l
i
o
T
u
r
n
o
v
e
r
": "
s
t
r
i
n
g
"
"Portfolio Turnover": "string"
},
"totalPerformance": {
"tradeStatistics": {
"startDateTime": "2021-11-26T15:18:27.693Z",
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
},
"closedTrades": [
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
]
},
"nodeName": "string",
"outOfSampleMaxEndDate": "2021-11-
26T15:18:27.693Z",
"outOfSampleDays": 0
}
ResearchGuide Model - A power gauge for backtests, time and parameters to estimate the overfitting risk.
minutes
integer
Number of minutes used in developing the current backtest.
backtestCount
integer
The quantity of backtests run in the project.
parameters
integer
Number of parameters detected.
Example
{
"minutes": 0,
"backtestCount": 0,
"parameters": 0
}
ChartSummary Model - Contains the names of all charts
name
string
Name of the Chart.
Example
{
"name": "string"
}
ParameterSet Model - Parameter set.
name
string
Name of parameter.
value
number
Value of parameter.
Example
{
"name": "string",
"value": 0
}
AlgorithmPerformance Model - The AlgorithmPerformance class is a wrapper for TradeStatistics and PortfolioStatistics.
tradeStatistics
TradeStatistics object
A set of statistics calculated from a list of closed trades.
portfolioStatistics
PortfolioStatistics object
Represents a set of statistics calculated from equity and
benchmark samples.
closedTrades
Trade Array
The algorithm statistics on portfolio.
{
"tradeStatistics": {
"startDateTime": "2021-11-26T15:18:27.693Z",
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
Example
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
},
"closedTrades": [
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
]
}
TradeStatistics Model - A set of statistics calculated from a list of closed trades.
startDateTime
string($date-time)
The entry date/time of the first trade.
endDateTime
string($date-time)
The exit date/time of the first trade.
totalNumberOfTrades
integer
The total number of trades.
numberOfWinningTrades
integer
The total number of winning trades.
numberOfLosingTrades
integer
The total number of losing trades.
totalProfitLoss
number
The total profit/loss for all trades (as symbol currency).
totalProfit
number
The total profit for all winning trades (as symbol currency).
totalLoss
number
The total loss for all losing trades (as symbol currency).
largestProfit
number
The largest profit in a single trade (as symbol currency).
largestLoss
number
The largest loss in a single trade (as symbol currency).
averageProfitLoss
number
The average profit/loss (a.k.a. Expectancy or Average
Trade) for all trades (as symbol currency).
averageProfit
number
The average profit for all winning trades (as symbol
currency).
averageLoss
number
The average loss for all winning trades (as symbol
currency).
averageTradeDuration
string
The average duration for all trades.
averageWinningTradeDuration
string
The average duration for all winning trades.
averageLosingTradeDuration
string
The average duration for all losing trades.
medianTradeDuration
string
The median duration for all trades.
medianWinningTradeDuration
string
The median duration for all winning trades.
medianLosingTradeDuration
string
The median duration for all losing trades.
maxConsecutiveWinningTrades
integer
The maximum number of consecutive winning trades.
maxConsecutiveLosingTrades
integer
The maximum number of consecutive losing trades.
profitLossRatio
number
The ratio of the average profit per trade to the average loss
per trade.
winLossRatio
number
The ratio of the number of winning trades to the number of
losing trades.
winRate
number
The ratio of the number of winning trades to the total
number of trades.
lossRate
number
The ratio of the number of losing trades to the total number
of trades.
averageMAE
number
The average Maximum Adverse Excursion for all trades.
averageMFE
number
The average Maximum Adverse Excursion for all trades.
largestMAE
number
The average Maximum Favorable Excursion for all trades.
largestMFE
number
The largest Maximum Adverse Excursion in a single trade
(as symbol currency).
maximumClosedTradeDrawdown
number
The maximum closed-trade drawdown for all trades (as
symbol currency).
maximumIntraTradeDrawdown
number
The maximum intra-trade drawdown for all trades (as
symbol currency).
profitLossStandardDeviation
number
The standard deviation of the profits/losses for all trades
(as symbol currency).
profitLossDownsideDeviation
number
The downside deviation of the profits/losses for all trades
(as symbol currency).
profitFactor
number
The ratio of the total profit to the total loss.
sharpeRatio
number
The ratio of the average profit/loss to the standard
deviation.
sortinoRatio
number
The ratio of the average profit/loss to the downside
deviation.
profitToMaxDrawdownRatio
number
The ratio of the total profit/loss to the maximum closed
trade drawdown.
maximumEndTradeDrawdown
number
The maximum amount of profit given back by a single trade
before exit (as symbol currency).
averageEndTradeDrawdown
number
The average amount of profit given back by all trades
before exit (as symbol currency).
maximumDrawdownDuration
string
The maximum amount of time to recover from a drawdown
(longest time between new equity highs or peaks).
totalFees
number
The sum of fees for all trades.
Example
{
"startDateTime": "2021-11-26T15:18:27.693Z",
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
}
PortfolioStatistics Model - Represents a set of statistics calculated from equity and benchmark samples.
averageWinRate
number
The average rate of return for winning trades.
averageLossRate
number
The average rate of return for losing trades.
profitLossRatio
number
The ratio of the average win rate to the average loss rate.
winRate
number
The ratio of the number of winning trades to the total
number of trades.
lossRate
number
The ratio of the number of losing trades to the total number
of trades.
expectancy
number
The expected value of the rate of return.
startEquity
number
Initial Equity Total Value.
endEquity
number
Final Equity Total Value.
compoundingAnnualReturn
number
Annual compounded returns statistic based on the finalstarting capital and years.
drawdown
number
Drawdown maximum percentage.
totalNetProfit
number
The total net profit percentage.
sharpeRatio
number
Sharpe ratio with respect to risk free rate: measures excess
of return per unit of risk.
probabilisticSharpeRatio
number
Probabilistic Sharpe Ratio is a probability measure
associated with the Sharpe ratio. It informs us of the
probability that the estimated Sharpe ratio is greater than a
chosen benchmark.
sortinoRatio
number
Sortino ratio with respect to risk free rate; measures excess
of return per unit of downside risk.
alpha
number
Algorithm "Alpha" statistic - abnormal returns over the risk
free rate and the relationshio (beta) with the benchmark
returns.
beta
number
Algorithm beta statistic - the covariance between the
algorithm and benchmark performance, divided by
benchmark variance.
annualStandardDeviation
number
Annualized standard deviation.
annualVariance
number
Annualized variance statistic calculation using the daily
performance variance and trading days per year.
informationRatio
number
Information ratio - risk adjusted return.
trackingError
number
Tracking error volatility (TEV) statistic - a measure of how
closely a portfolio follows the index to which it is
benchmarked.
treynorRatio
number
Treynor ratio statistic is a measurement of the returns
earned in excess of that which could have been earned on
an investment that has no diversifiable risk.
portfolioTurnover
number
The average Portfolio Turnover.
valueAtRisk99
number
The 1-day VaR for the portfolio, using the Variancecovariance approach. Assumes a 99% confidence level, 1
year lookback period, and that the returns are normally
distributed.
valueAtRisk95
number
The 1-day VaR for the portfolio, using the Variancecovariance approach. Assumes a 95% confidence level, 1
year lookback period, and that the returns are normally
distributed.
Example
{
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
}
Trade Model - Represents a closed trade.
symbol
Symbol object
Represents a unique security identifier. This is made of two
components, the unique SID and the Value. The value is the
current ticker symbol while the SID is constant over the life
of a security.
entryTime
string($date-time)
The date and time the trade was opened.
entryPrice
number
The price at which the trade was opened (or the average
price if multiple entries).
direction
string Enum
Direction of a trade. Options : ['Long', 'Short']
quantity
number
The total unsigned quantity of the trade.
exitTime
string($date-time)
The date and time the trade was closed.
exitPrice
number
The price at which the trade was closed (or the average
price if multiple exits).
profitLoss
number
The gross profit/loss of the trade (as account currency).
totalFees
number
The total fees associated with the trade (always positive
value) (as account currency).
mae
number
The Maximum Adverse Excursion (as account currency).
mfe
number
The Maximum Favorable Excursion (as account currency).
duration
string
The duration of the trade.
endTradeDrawdown
number
The amount of profit given back before the trade was
closed.
Example
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
Symbol Model - Represents a unique security identifier. This is made of two components, the unique SID and the Value. The
value is the current ticker symbol while the SID is constant over the life of a security.
value
string
The current symbol for this ticker.
id
string
The security identifier for this symbol.
permtick
string
The current symbol for this ticker.
Example
{
"value": "string",
"id": "string",
"permtick": "string"
}
RuntimeStatistics Model
Equity
string
example:
$100.00
Total portfolio value.
Fees
string
example: -$100.00
Transaction fee.
Holdings
string
example:
$100.00
Equity value of security holdings.
Net Profit
string
example:
$100.00
Net profit.
Probabilistic Sharpe Ratio
string
example: 50.00%
Probabilistic Sharpe Ratio.
Return
string
example: 50.00%
Return.
Unrealized
string
example:
$100.00
Unrealized profit/loss.
Volume
string
example:
$100.00
Total transaction volume.
Example
{
"Equity": "$100.00",
"Fees": "-$100.00",
"Holdings": "$100.00",
"Net Profit": "$100.00",
"Probabilistic Sharpe Ratio": "50.00%",
"Return": "50.00%",
"Unrealized": "$100.00",
"Volume": "$100.00"
}
StatisticsResult Model - Statistics information sent during the algorithm operations.
Total Orders
string
Total nuber of orders.
Average Win
string
The average rate of return for winning trades.
Average Loss
string
The average rate of return for losing trades.
Compounding Annual Return
string
Annual compounded returns statistic based on the finalstarting capital and years.
Drawdown
string
Drawdown maximum percentage.
Expectancy
string
The expected value of the rate of return.
Start Equity
string
Initial Equity Total Value.
End Equity
string
Final Equity Total Value.
Net Profit
string
The total net profit percentage.
Sharpe Ratio
string
Sharpe ratio with respect to risk free rate; measures excess
of return per unit of risk.
Sortino Ratio
string
Sortino ratio with respect to risk free rate; measures excess
of return per unit of downside risk.
Probabilistic Sharpe Ratio
string
Is a probability measure associated with the Sharpe ratio. It
informs us of the probability that the estimated Sharpe ratio
is greater than a chosen benchmark.
Loss Rate
string
The ratio of the number of losing trades to the total number
of trades.
Win Rate
string
The ratio of the number of winning trades to the total
number of trades.
Profit-Loss Ratio
string
The ratio of the average win rate to the average loss rate.
Alpha
string
Algorithm "Alpha" statistic - abnormal returns over the risk
free rate and the relationshio (beta) with the benchmark
returns.
Beta
string
Algorithm "beta" statistic - the covariance between the
algorithm and benchmark performance, divided by
benchmark's variance.
Annual Standard Deviation
string
Annualized standard deviation.
Annual Variance
string
Annualized variance statistic calculation using the daily
performance variance and trading days per year.
Information Ratio
string
Information ratio - risk adjusted return.
Tracking Error
string
Tracking error volatility (TEV) statistic - a measure of how
closely a portfolio follows the index to which it is
benchmarked.
Treynor Ratio
string
Treynor ratio statistic is a measurement of the returns
earned in excess of that which could have been earned on
an investment that has no diversifiable risk.
Total Fees
string
Total amount of fees.
Estimated Strategy Capacity
string
The estimated total capacity of the strategy at a point in
time.
Lowest Capacity Asset
string
Provide a reference to the lowest capacity symbol used in
scaling down the capacity for debugging.
Portfolio Turnover
string
The average Portfolio Turnover.
Example
{
"Total Orders": "string",
"Average Win": "string",
"Average Loss": "string",
"Compounding Annual Return": "string",
"Drawdown": "string",
"Expectancy": "string",
"Start Equity": "string",
"End Equity": "string",
"Net Profit": "string",
"Sharpe Ratio": "string",
"Sortino Ratio": "string",
"Probabilistic Sharpe Ratio": "string",
"Loss Rate": "string",
"Win Rate": "string",
"Profit-Loss Ratio": "string",
"Alpha": "string",
"Beta": "string",
"Annual Standard Deviation": "string",
"Annual Variance": "string",
"Information Ratio": "string",
"Tracking Error": "string",
"Treynor Ratio": "string",
"Total Fees": "string",
"Estimated Strategy Capacity": "string",
"Lowest Capacity Asset": "string",
"Portfolio Turnover": "string"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > Read Backtest
Backtest Management
Read Backtest
The QuantConnect REST API lets you read backtest results from our cloud servers through URL endpoints.
Backtest Statistics
Charts
Orders
Insights
API Reference > Backtest Management > Read Backtest > Backtest Statistics
Read Backtest
Backtest Statistics
Introduction
Read out that backtest from the project (optionally with the charts included).
Request
Fetch the results for the project Id and backtest Id provided (and optional chart name provided). The /backtests/read API
accepts requests in the following format:
ReadBacktestRequest Model - Request to read a single backtest from a project.
projectId
integer
Id of the project from which to read one or multiple
backtests.
backtestId
string
When provided, specific backtest Id to read.
chart
string
Optional. If provided, the API will return the backtests
charts.
Example
{
"projectId": 0,
"backtestId": "string",
"chart": "string"
}
Responses
The /backtests/read API provides a response in the following format:
200 Success
BacktestResponse Model - Collection container for a list of backtests for a project.
backtest
BacktestResult Array
Collection of backtests for a project.
debugging
boolean
Indicates if the backtest is run under debugging mode.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
{
"backtest": [
{
"note": "string",
"name": "string",
"organizationId": 0,
"projectId": 0,
"completed": true,
"optimizationId": "string",
"backtestId": "string",
"tradeableDates": 0,
"researchGuide": {
"minutes": 0,
"backtestCount": 0,
"parameters": 0
},
"backtestStart": "2021-11-26T15:18:27.693Z",
"backtestEnd": "2021-11-26T15:18:27.693Z",
"created": "2021-11-26T15:18:27.693Z",
"snapshotId": 0,
"status": "Completed.",
"error": "string",
"stacktrace": "string",
"progress": 0,
"hasInitializeError": true,
"charts": {
"name": "string"
},
"parameterSet": {
"name": "string",
"value": 0
},
"rollingWindow": {
"tradeStatistics": {
"startDateTime": "2021-11-
26T15:18:27.693Z",
"endDateTime": "2021-11-
26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
E
x
a
m
ple
"
a
v
e
r
a
g
e
W
i
n
R
a
t
e
": 0
,
"
a
v
e
r
a
g
e
L
o
s
s
R
a
t
e
": 0
,
"
p
r
o
f
i
t
L
o
s
s
R
a
t
i
o
": 0
,
"
w
i
n
R
a
t
e
": 0
,
"
l
o
s
s
R
a
t
e
": 0
,
"
e
x
p
e
c
t
a
n
c
y
": 0
,
"
s
t
a
r
t
E
q
u
i
t
y
": 0
,
"
e
n
d
E
q
u
i
t
y
": 0
,
"
c
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": 0
,
"
d
r
a
w
d
o
w
n
": 0
,
"
t
o
t
a
l
N
e
t
P
r
o
f
i
t
": 0
,
"
s
h
a
r
p
e
R
a
t
i
o
": 0
,
"
p
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": 0
,
"
s
o
r
t
i
n
o
R
a
t
i
o
": 0
,
"
a
l
p
h
a
": 0
,
"
b
e
t
a
": 0
,
"
a
n
n
u
a
l
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": 0
,
"
a
n
n
u
a
l
V
a
r
i
a
n
c
e
": 0
,
"
i
n
f
o
r
m
a
t
i
o
n
R
a
t
i
o
": 0
,
"
t
r
a
c
k
i
n
g
E
r
r
o
r
": 0
,
"
t
r
e
y
n
o
r
R
a
t
i
o
": 0
,
"
p
o
r
t
f
o
l
i
o
T
u
r
n
o
v
e
r
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
9
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
5
": 0
}
,
"
c
l
o
s
e
d
T
r
a
d
e
s
": [
{
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
e
n
t
r
y
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
n
t
r
y
P
r
i
c
e
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
L
o
n
g
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
e
x
i
t
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
x
i
t
P
r
i
c
e
": 0
,
"
p
r
o
f
i
t
L
o
s
s
": 0
,
"
t
o
t
a
l
F
e
e
s
": 0
,
"
m
a
e
": 0
,
"
m
f
e
": 0
,
"
d
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
e
n
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
}
]
}
,
"
r
u
n
t
i
m
e
S
t
a
t
i
s
t
i
c
s
": {
"
E
q
u
i
t
y
": "
$
1
0
0.0
0
"
,
"
F
e
e
s
": "
-
$
1
0
0.0
0
"
,
"
H
o
l
d
i
n
g
s
": "
$
1
0
0.0
0
"
,
"
N
e
t
P
r
o
f
i
t
": "
$
1
0
0.0
0
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
5
0.0
0
%
"
,
"
R
e
t
u
r
n
": "
5
0.0
0
%
"
,
"
U
n
r
e
a
l
i
z
e
d
": "
$
1
0
0.0
0
"
,
"
V
o
l
u
m
e
": "
$
1
0
0.0
0
"
}
,
"
s
t
a
t
i
s
t
i
c
s
": {
"
T
o
t
a
l
O
r
d
e
r
s
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
W
i
n
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
L
o
s
s
": "
s
t
r
i
n
g
"
,
"
C
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": "
s
t
r
i
n
g
"
,
"
D
r
a
w
d
o
w
n
": "
s
t
r
i
n
g
"
,
"
E
x
p
e
c
t
a
n
c
y
": "
s
t
r
i
n
g
"
,
"
S
t
a
r
t
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
E
n
d
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
N
e
t
P
r
o
f
i
t
": "
s
t
r
i
n
g
"
,
"
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
S
o
r
t
i
n
o
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
L
o
s
s
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
W
i
n
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
P
r
o
f
i
t
-
L
o
s
s
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
A
l
p
h
a
": "
s
t
r
i
n
g
"
,
"
B
e
t
a
": "
s
t
r
i
n
g
"
,
"
A
n
n
u
a
l
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
A
n
n
u
a
l
V
a
r
i
a
n
c
e
": "
s
t
r
i
n
g
"
,
"
I
n
f
o
r
m
a
t
i
o
n
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
T
r
a
c
k
i
n
g
E
r
r
o
r
": "
s
t
r
i
n
g
"
,
"
T
r
e
y
n
o
r
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"Treynor Ratio": "string",
"Total Fees": "string",
"Estimated Strategy Capacity": "string",
"Lowest Capacity Asset": "string",
"Portfolio Turnover": "string"
},
"totalPerformance": {
"tradeStatistics": {
"startDateTime": "2021-11-
26T15:18:27.693Z",
"endDateTime": "2021-11-
26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
},
"closedTrades": [
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-
26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-
26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
]
},
"nodeName": "string",
"outOfSampleMaxEndDate": "2021-11-
26T15:18:27.693Z",
"outOfSampleDays": 0
}
],
"debugging": true,
"success": true,
"errors": [
"string"
]
}
BacktestResult Model - Results object class. Results are exhaust from backtest or live algorithms running in LEAN.
note
string
Note on the backtest attached by the user.
name
string
Name of the backtest.
organizationId
integer
Organization ID.
projectId
integer
Project ID.
completed
boolean
Boolean true when the backtest is completed.
optimizationId
string
Optimization task ID, if the backtest is part of an
optimization.
backtestId
string
Assigned backtest ID.
tradeableDates
integer
Number of traadeable days.
researchGuide
ResearchGuide object
A power gauge for backtests, time and parameters to
estimate the overfitting risk.
backtestStart
string($date-time)
The starting time of the backtest.
backtestEnd
string($date-time)
The ending time of the backtest.
created
string($date-time)
Backtest creation date and time.
snapshotId
integer
Snapshot id of this backtest result.
status
string Enum
Status of the backtest. Options : ['Completed.', 'In
Queue...', "'Running: _%'"]
error
string
Backtest error message.
stacktrace
string
Backtest error stacktrace.
progress
number
Progress of the backtest in percent 0-1.
hasInitializeError
boolean
Indicates if the backtest has error during initialization.
charts
ChartSummary object
Charts updates for the live algorithm since the last result
packet.
parameterSet
ParameterSet object
Parameters used in the backtest.
rollingWindow
AlgorithmPerformance object
Rolling window detailed statistics.
runtimeStatistics
RuntimeStatistics object
Runtime banner/updating statistics in the title banner of the
live algorithm GUI.
statistics
StatisticsResult object
Statistics information sent during the algorithm operations.
totalPerformance
AlgorithmPerformance object
The algorithm performance statistics.
nodeName
string
The backtest node name.
outOfSampleMaxEndDate
string($date-time)
End date of out of sample data.
outOfSampleDays
integer
Number of days of out of sample days.
{
"note": "string",
"note": "string",
"name": "string",
"organizationId": 0,
"projectId": 0,
"completed": true,
"optimizationId": "string",
"backtestId": "string",
"tradeableDates": 0,
"researchGuide": {
"minutes": 0,
"backtestCount": 0,
"parameters": 0
},
"backtestStart": "2021-11-26T15:18:27.693Z",
"backtestEnd": "2021-11-26T15:18:27.693Z",
"created": "2021-11-26T15:18:27.693Z",
"snapshotId": 0,
"status": "Completed.",
"error": "string",
"stacktrace": "string",
"progress": 0,
"hasInitializeError": true,
"charts": {
"name": "string"
},
"parameterSet": {
"name": "string",
"value": 0
},
"rollingWindow": {
"tradeStatistics": {
"startDateTime": "2021-11-26T15:18:27.693Z",
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
E
x
a
m
ple
"
e
n
d
E
q
u
i
t
y
": 0
,
"
c
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": 0
,
"
d
r
a
w
d
o
w
n
": 0
,
"
t
o
t
a
l
N
e
t
P
r
o
f
i
t
": 0
,
"
s
h
a
r
p
e
R
a
t
i
o
": 0
,
"
p
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": 0
,
"
s
o
r
t
i
n
o
R
a
t
i
o
": 0
,
"
a
l
p
h
a
": 0
,
"
b
e
t
a
": 0
,
"
a
n
n
u
a
l
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": 0
,
"
a
n
n
u
a
l
V
a
r
i
a
n
c
e
": 0
,
"
i
n
f
o
r
m
a
t
i
o
n
R
a
t
i
o
": 0
,
"
t
r
a
c
k
i
n
g
E
r
r
o
r
": 0
,
"
t
r
e
y
n
o
r
R
a
t
i
o
": 0
,
"
p
o
r
t
f
o
l
i
o
T
u
r
n
o
v
e
r
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
9
": 0
,
"
v
a
l
u
e
A
t
R
i
s
k
9
5
": 0
}
,
"
c
l
o
s
e
d
T
r
a
d
e
s
": [
{
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
e
n
t
r
y
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
n
t
r
y
P
r
i
c
e
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
L
o
n
g
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
e
x
i
t
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
x
i
t
P
r
i
c
e
": 0
,
"
p
r
o
f
i
t
L
o
s
s
": 0
,
"
t
o
t
a
l
F
e
e
s
": 0
,
"
m
a
e
": 0
,
"
m
f
e
": 0
,
"
d
u
r
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
e
n
d
T
r
a
d
e
D
r
a
w
d
o
w
n
": 0
}
]
}
,
"
r
u
n
t
i
m
e
S
t
a
t
i
s
t
i
c
s
": {
"
E
q
u
i
t
y
": "
$
1
0
0.0
0
"
,
"
F
e
e
s
": "
-
$
1
0
0.0
0
"
,
"
H
o
l
d
i
n
g
s
": "
$
1
0
0.0
0
"
,
"
N
e
t
P
r
o
f
i
t
": "
$
1
0
0.0
0
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
5
0.0
0
%
"
,
"
R
e
t
u
r
n
": "
5
0.0
0
%
"
,
"
U
n
r
e
a
l
i
z
e
d
": "
$
1
0
0.0
0
"
,
"
V
o
l
u
m
e
": "
$
1
0
0.0
0
"
}
,
"
s
t
a
t
i
s
t
i
c
s
": {
"
T
o
t
a
l
O
r
d
e
r
s
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
W
i
n
": "
s
t
r
i
n
g
"
,
"
A
v
e
r
a
g
e
L
o
s
s
": "
s
t
r
i
n
g
"
,
"
C
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": "
s
t
r
i
n
g
"
,
"
D
r
a
w
d
o
w
n
": "
s
t
r
i
n
g
"
,
"
E
x
p
e
c
t
a
n
c
y
": "
s
t
r
i
n
g
"
,
"
S
t
a
r
t
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
E
n
d
E
q
u
i
t
y
": "
s
t
r
i
n
g
"
,
"
N
e
t
P
r
o
f
i
t
": "
s
t
r
i
n
g
"
,
"
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
S
o
r
t
i
n
o
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
L
o
s
s
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
W
i
n
R
a
t
e
": "
s
t
r
i
n
g
"
,
"
P
r
o
f
i
t
-
L
o
s
s
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
A
l
p
h
a
": "
s
t
r
i
n
g
"
,
"
B
e
t
a
": "
s
t
r
i
n
g
"
,
"
A
n
n
u
a
l
S
t
a
n
d
a
r
d
D
e
v
i
a
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
A
n
n
u
a
l
V
a
r
i
a
n
c
e
": "
s
t
r
i
n
g
"
,
"
I
n
f
o
r
m
a
t
i
o
n
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
T
r
a
c
k
i
n
g
E
r
r
o
r
": "
s
t
r
i
n
g
"
,
"
T
r
e
y
n
o
r
R
a
t
i
o
": "
s
t
r
i
n
g
"
,
"
T
o
t
a
l
F
e
e
s
": "
s
t
r
i
n
g
"
,
"
E
s
t
i
m
a
t
e
d
S
t
r
a
t
e
g
y
C
a
p
a
c
i
t
y
": "
s
t
r
i
n
g
"
,
"
L
o
w
e
s
t
C
a
p
a
c
i
t
y
A
s
s
e
t
": "
s
t
r
i
n
g
"
,
"
P
o
r
t
f
o
l
i
o
T
u
r
n
o
v
e
r
": "
s
t
r
i
n
g
"
}
,
"
t
o
t
a
l
P
e
r
f
o
r
m
a
n
c
e
": {
"
t
r
a
d
e
S
t
a
t
i
s
t
i
c
s
": {
"
s
t
a
r
t
D
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
e
n
d
D
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
},
"closedTrades": [
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
]
},
"nodeName": "string",
"outOfSampleMaxEndDate": "2021-11-
26T15:18:27.693Z",
"outOfSampleDays": 0
}
ResearchGuide Model - A power gauge for backtests, time and parameters to estimate the overfitting risk.
minutes
integer
Number of minutes used in developing the current backtest.
backtestCount
integer
The quantity of backtests run in the project.
parameters
integer
Number of parameters detected.
Example
{
"minutes": 0,
"backtestCount": 0,
"parameters": 0
}
ChartSummary Model - Contains the names of all charts
name
string
Name of the Chart.
Example
{
"name": "string"
}
ParameterSet Model - Parameter set.
name
string
Name of parameter.
value
number
Value of parameter.
Example
{
"name": "string",
"value": 0
}
AlgorithmPerformance Model - The AlgorithmPerformance class is a wrapper for TradeStatistics and PortfolioStatistics.
tradeStatistics
TradeStatistics object
A set of statistics calculated from a list of closed trades.
portfolioStatistics
PortfolioStatistics object
Represents a set of statistics calculated from equity and
benchmark samples.
closedTrades
Trade Array
The algorithm statistics on portfolio.
{
"tradeStatistics": {
"startDateTime": "2021-11-26T15:18:27.693Z",
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
Example
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
},
"portfolioStatistics": {
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
},
"closedTrades": [
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
]
}
TradeStatistics Model - A set of statistics calculated from a list of closed trades.
startDateTime
string($date-time)
The entry date/time of the first trade.
endDateTime
string($date-time)
The exit date/time of the first trade.
totalNumberOfTrades
integer
The total number of trades.
numberOfWinningTrades
integer
The total number of winning trades.
numberOfLosingTrades
integer
The total number of losing trades.
totalProfitLoss
number
The total profit/loss for all trades (as symbol currency).
totalProfit
number
The total profit for all winning trades (as symbol currency).
totalLoss
number
The total loss for all losing trades (as symbol currency).
largestProfit
number
The largest profit in a single trade (as symbol currency).
largestLoss
number
The largest loss in a single trade (as symbol currency).
averageProfitLoss
number
The average profit/loss (a.k.a. Expectancy or Average
Trade) for all trades (as symbol currency).
averageProfit
number
The average profit for all winning trades (as symbol
currency).
averageLoss
number
The average loss for all winning trades (as symbol
currency).
averageTradeDuration
string
The average duration for all trades.
averageWinningTradeDuration
string
The average duration for all winning trades.
averageLosingTradeDuration
string
The average duration for all losing trades.
medianTradeDuration
string
The median duration for all trades.
medianWinningTradeDuration
string
The median duration for all winning trades.
medianLosingTradeDuration
string
The median duration for all losing trades.
maxConsecutiveWinningTrades
integer
The maximum number of consecutive winning trades.
maxConsecutiveLosingTrades
integer
The maximum number of consecutive losing trades.
profitLossRatio
number
The ratio of the average profit per trade to the average loss
per trade.
winLossRatio
number
The ratio of the number of winning trades to the number of
losing trades.
winRate
number
The ratio of the number of winning trades to the total
number of trades.
lossRate
number
The ratio of the number of losing trades to the total number
of trades.
averageMAE
number
The average Maximum Adverse Excursion for all trades.
averageMFE
number
The average Maximum Adverse Excursion for all trades.
largestMAE
number
The average Maximum Favorable Excursion for all trades.
largestMFE
number
The largest Maximum Adverse Excursion in a single trade
(as symbol currency).
maximumClosedTradeDrawdown
number
The maximum closed-trade drawdown for all trades (as
symbol currency).
maximumIntraTradeDrawdown
number
The maximum intra-trade drawdown for all trades (as
symbol currency).
profitLossStandardDeviation
number
The standard deviation of the profits/losses for all trades
(as symbol currency).
profitLossDownsideDeviation
number
The downside deviation of the profits/losses for all trades
(as symbol currency).
profitFactor
number
The ratio of the total profit to the total loss.
sharpeRatio
number
The ratio of the average profit/loss to the standard
deviation.
sortinoRatio
number
The ratio of the average profit/loss to the downside
deviation.
profitToMaxDrawdownRatio
number
The ratio of the total profit/loss to the maximum closed
trade drawdown.
maximumEndTradeDrawdown
number
The maximum amount of profit given back by a single trade
before exit (as symbol currency).
averageEndTradeDrawdown
number
The average amount of profit given back by all trades
before exit (as symbol currency).
maximumDrawdownDuration
string
The maximum amount of time to recover from a drawdown
(longest time between new equity highs or peaks).
totalFees
number
The sum of fees for all trades.
Example
{
"startDateTime": "2021-11-26T15:18:27.693Z",
"endDateTime": "2021-11-26T15:18:27.693Z",
"totalNumberOfTrades": 0,
"numberOfWinningTrades": 0,
"numberOfLosingTrades": 0,
"totalProfitLoss": 0,
"totalProfit": 0,
"totalLoss": 0,
"largestProfit": 0,
"largestLoss": 0,
"averageProfitLoss": 0,
"averageProfit": 0,
"averageLoss": 0,
"averageTradeDuration": "string",
"averageWinningTradeDuration": "string",
"averageLosingTradeDuration": "string",
"medianTradeDuration": "string",
"medianWinningTradeDuration": "string",
"medianLosingTradeDuration": "string",
"maxConsecutiveWinningTrades": 0,
"maxConsecutiveLosingTrades": 0,
"profitLossRatio": 0,
"winLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"averageMAE": 0,
"averageMFE": 0,
"largestMAE": 0,
"largestMFE": 0,
"maximumClosedTradeDrawdown": 0,
"maximumIntraTradeDrawdown": 0,
"profitLossStandardDeviation": 0,
"profitLossDownsideDeviation": 0,
"profitFactor": 0,
"sharpeRatio": 0,
"sortinoRatio": 0,
"profitToMaxDrawdownRatio": 0,
"maximumEndTradeDrawdown": 0,
"averageEndTradeDrawdown": 0,
"maximumDrawdownDuration": "string",
"totalFees": 0
}
PortfolioStatistics Model - Represents a set of statistics calculated from equity and benchmark samples.
averageWinRate
number
The average rate of return for winning trades.
averageLossRate
number
The average rate of return for losing trades.
profitLossRatio
number
The ratio of the average win rate to the average loss rate.
winRate
number
The ratio of the number of winning trades to the total
number of trades.
lossRate
number
The ratio of the number of losing trades to the total number
of trades.
expectancy
number
The expected value of the rate of return.
startEquity
number
Initial Equity Total Value.
endEquity
number
Final Equity Total Value.
compoundingAnnualReturn
number
Annual compounded returns statistic based on the finalstarting capital and years.
drawdown
number
Drawdown maximum percentage.
totalNetProfit
number
The total net profit percentage.
sharpeRatio
number
Sharpe ratio with respect to risk free rate: measures excess
of return per unit of risk.
probabilisticSharpeRatio
number
Probabilistic Sharpe Ratio is a probability measure
associated with the Sharpe ratio. It informs us of the
probability that the estimated Sharpe ratio is greater than a
chosen benchmark.
sortinoRatio
number
Sortino ratio with respect to risk free rate; measures excess
of return per unit of downside risk.
alpha
number
Algorithm "Alpha" statistic - abnormal returns over the risk
free rate and the relationshio (beta) with the benchmark
returns.
beta
number
Algorithm beta statistic - the covariance between the
algorithm and benchmark performance, divided by
benchmark variance.
annualStandardDeviation
number
Annualized standard deviation.
annualVariance
number
Annualized variance statistic calculation using the daily
performance variance and trading days per year.
informationRatio
number
Information ratio - risk adjusted return.
trackingError
number
Tracking error volatility (TEV) statistic - a measure of how
closely a portfolio follows the index to which it is
benchmarked.
treynorRatio
number
Treynor ratio statistic is a measurement of the returns
earned in excess of that which could have been earned on
an investment that has no diversifiable risk.
portfolioTurnover
number
The average Portfolio Turnover.
valueAtRisk99
number
The 1-day VaR for the portfolio, using the Variancecovariance approach. Assumes a 99% confidence level, 1
year lookback period, and that the returns are normally
distributed.
valueAtRisk95
number
The 1-day VaR for the portfolio, using the Variancecovariance approach. Assumes a 95% confidence level, 1
year lookback period, and that the returns are normally
distributed.
Example
{
"averageWinRate": 0,
"averageLossRate": 0,
"profitLossRatio": 0,
"winRate": 0,
"lossRate": 0,
"expectancy": 0,
"startEquity": 0,
"endEquity": 0,
"compoundingAnnualReturn": 0,
"drawdown": 0,
"totalNetProfit": 0,
"sharpeRatio": 0,
"probabilisticSharpeRatio": 0,
"sortinoRatio": 0,
"alpha": 0,
"beta": 0,
"annualStandardDeviation": 0,
"annualVariance": 0,
"informationRatio": 0,
"trackingError": 0,
"treynorRatio": 0,
"portfolioTurnover": 0,
"valueAtRisk99": 0,
"valueAtRisk95": 0
}
Trade Model - Represents a closed trade.
symbol
Symbol object
Represents a unique security identifier. This is made of two
components, the unique SID and the Value. The value is the
current ticker symbol while the SID is constant over the life
of a security.
entryTime
string($date-time)
The date and time the trade was opened.
entryPrice
number
The price at which the trade was opened (or the average
price if multiple entries).
direction
string Enum
Direction of a trade. Options : ['Long', 'Short']
quantity
number
The total unsigned quantity of the trade.
exitTime
string($date-time)
The date and time the trade was closed.
exitPrice
number
The price at which the trade was closed (or the average
price if multiple exits).
profitLoss
number
The gross profit/loss of the trade (as account currency).
totalFees
number
The total fees associated with the trade (always positive
value) (as account currency).
mae
number
The Maximum Adverse Excursion (as account currency).
mfe
number
The Maximum Favorable Excursion (as account currency).
duration
string
The duration of the trade.
endTradeDrawdown
number
The amount of profit given back before the trade was
closed.
Example
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"entryTime": "2021-11-26T15:18:27.693Z",
"entryPrice": 0,
"direction": "Long",
"quantity": 0,
"exitTime": "2021-11-26T15:18:27.693Z",
"exitPrice": 0,
"profitLoss": 0,
"totalFees": 0,
"mae": 0,
"mfe": 0,
"duration": "string",
"endTradeDrawdown": 0
}
Symbol Model - Represents a unique security identifier. This is made of two components, the unique SID and the Value. The
value is the current ticker symbol while the SID is constant over the life of a security.
value
string
The current symbol for this ticker.
id
string
The security identifier for this symbol.
permtick
string
The current symbol for this ticker.
Example
{
"value": "string",
"id": "string",
"permtick": "string"
}
RuntimeStatistics Model
Equity
string
example:
$100.00
Total portfolio value.
Fees
string
example: -$100.00
Transaction fee.
Holdings
string
example:
$100.00
Equity value of security holdings.
Net Profit
string
example:
$100.00
Net profit.
Probabilistic Sharpe Ratio
string
example: 50.00%
Probabilistic Sharpe Ratio.
Return
string
example: 50.00%
Return.
Unrealized
string
example:
$100.00
Unrealized profit/loss.
Volume
string
example:
$100.00
Total transaction volume.
Example
{
"Equity": "$100.00",
"Fees": "-$100.00",
"Holdings": "$100.00",
"Net Profit": "$100.00",
"Probabilistic Sharpe Ratio": "50.00%",
"Return": "50.00%",
"Unrealized": "$100.00",
"Volume": "$100.00"
}
StatisticsResult Model - Statistics information sent during the algorithm operations.
Total Orders
string
Total nuber of orders.
Average Win
string
The average rate of return for winning trades.
Average Loss
string
The average rate of return for losing trades.
Compounding Annual Return
string
Annual compounded returns statistic based on the finalstarting capital and years.
Drawdown
string
Drawdown maximum percentage.
Expectancy
string
The expected value of the rate of return.
Start Equity
string
Initial Equity Total Value.
End Equity
string
Final Equity Total Value.
Net Profit
string
The total net profit percentage.
Sharpe Ratio
string
Sharpe ratio with respect to risk free rate; measures excess
of return per unit of risk.
Sortino Ratio
string
Sortino ratio with respect to risk free rate; measures excess
of return per unit of downside risk.
Probabilistic Sharpe Ratio
string
Is a probability measure associated with the Sharpe ratio. It
informs us of the probability that the estimated Sharpe ratio
is greater than a chosen benchmark.
Loss Rate
string
The ratio of the number of losing trades to the total number
of trades.
Win Rate
string
The ratio of the number of winning trades to the total
number of trades.
Profit-Loss Ratio
string
The ratio of the average win rate to the average loss rate.
Alpha
string
Algorithm "Alpha" statistic - abnormal returns over the risk
free rate and the relationshio (beta) with the benchmark
returns.
Beta
string
Algorithm "beta" statistic - the covariance between the
algorithm and benchmark performance, divided by
benchmark's variance.
Annual Standard Deviation
string
Annualized standard deviation.
Annual Variance
string
Annualized variance statistic calculation using the daily
performance variance and trading days per year.
Information Ratio
string
Information ratio - risk adjusted return.
Tracking Error
string
Tracking error volatility (TEV) statistic - a measure of how
closely a portfolio follows the index to which it is
benchmarked.
Treynor Ratio
string
Treynor ratio statistic is a measurement of the returns
earned in excess of that which could have been earned on
an investment that has no diversifiable risk.
Total Fees
string
Total amount of fees.
Estimated Strategy Capacity
string
The estimated total capacity of the strategy at a point in
time.
Lowest Capacity Asset
string
Provide a reference to the lowest capacity symbol used in
scaling down the capacity for debugging.
Portfolio Turnover
string
The average Portfolio Turnover.
Example
{
"Total Orders": "string",
"Average Win": "string",
"Average Loss": "string",
"Compounding Annual Return": "string",
"Drawdown": "string",
"Expectancy": "string",
"Start Equity": "string",
"End Equity": "string",
"Net Profit": "string",
"Sharpe Ratio": "string",
"Sortino Ratio": "string",
"Probabilistic Sharpe Ratio": "string",
"Loss Rate": "string",
"Win Rate": "string",
"Profit-Loss Ratio": "string",
"Alpha": "string",
"Beta": "string",
"Annual Standard Deviation": "string",
"Annual Variance": "string",
"Information Ratio": "string",
"Tracking Error": "string",
"Treynor Ratio": "string",
"Total Fees": "string",
"Estimated Strategy Capacity": "string",
"Lowest Capacity Asset": "string",
"Portfolio Turnover": "string"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > Read Backtest > Charts
Read Backtest
Charts
Introduction
Read chart from a backtest.
Request
Request body to obtain a chart from a backtest. The /backtests/chart/read API accepts requests in the following format:
ReadBacktestChartRequest Model - Request body to obtain a chart from a backtest.
projectId
integer
example: 12345678
Project ID of the request.
backtestId
string
example: 2a748c241eb93b0b57b4747b3dacc80e
Associated Backtest ID for this chart request.
name
string
example: Strategy Equity
The requested chart name.
count
integer
example: 100
The number of data points to request.
start
integer
example: 1717801200
Optional. If provided, the Utc start seconds timestamp of
the request.
end
integer
example: 1743462000
Optional. If provided, the Utc end seconds timestamp of the
request.
Example
{
"projectId": 12345678,
"backtestId":
"2a748c241eb93b0b57b4747b3dacc80e",
"name": "Strategy Equity",
"count": 100,
"start": 1717801200,
"end": 1743462000
}
Responses
The /backtests/chart/read API provides a response in the following format:
200 Success
LoadingChartResponse Model - Response when the requested chart is being generated.
progress
number
Loading percentage of the chart generation process.
status
string
example:
loading
Status of the chart generation process.
success
boolean
Indicate if the API request was successful.
Example
{
"progress": 0,
"status": "loading",
"success": true
}
ReadChartResponse Model - Response with the requested chart from a live algorithm or backtest.
chart
Chart object
Single Parent Chart Object for Custom Charting.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"chart": {
"name": "string",
"chartType": "Overlay",
"series": {
"name": "string",
"unit": "string",
"index": 0,
"values": [
"object"
],
"seriesType": "Line",
"color": "string",
"scatterMarkerSymbol": "none"
}
},
"success": true,
"errors": [
"string"
]
}
Chart Model - Single Parent Chart Object for Custom Charting.
name
string
Name of the Chart.
chartType
string Enum
Type of the Chart, Overlayed or Stacked. Options :
['Overlay', 'Stacked']
series
Series object
List of Series Objects for this Chart.
Example
{
"name": "string",
"chartType": "Overlay",
"series": {
"name": "string",
"unit": "string",
"index": 0,
"values": [
"object"
],
"seriesType": "Line",
"color": "string",
"scatterMarkerSymbol": "none"
}
}
Series Model - Chart Series Object - Series data and properties for a chart.
name
string
Name of the series.
unit
string
Axis for the chart series.
index
integer
Index/position of the series on the chart.
values
object Array
Values for the series plot. These values are assumed to be
in ascending time order (first points earliest, last points
latest).
seriesType
string Enum
Chart type for the series. Options : ['Line', 'Scatter',
'Candle', 'Bar', 'Flag', 'StackedArea', 'Pie', 'Treemap']
color
string
Color the series.
scatterMarkerSymbol
string Enum
Shape or symbol for the marker in a scatter plot. Options :
['none', 'circle', 'square', 'diamond', 'triangle', 'triangledown']
Example
{
"name": "string",
"unit": "string",
"index": 0,
"values": [
"object"
],
"seriesType": "Line",
"color": "string",
"scatterMarkerSymbol": "none"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > Read Backtest > Orders
Read Backtest
Orders
Introduction
Read out the orders of a backtest.
Request
Fetch the orders of a backtest for the project Id, backtest Id and steps provided. The /backtests/orders/read API accepts
requests in the following format:
ReadBacktestOrdersRequest Model - Request to read orders from a backtest.
start
integer
Starting index of the orders to be fetched. Required if end >
100.
end
integer
Last index of the orders to be fetched. Note that end - start
must be less than 100.
projectId
integer
Id of the project from which to read the backtest.
backtestId
string
Id of the backtest from which to read the orders.
Example
{
"start": 0,
"end": 0,
"projectId": 0,
"backtestId": "string"
}
Responses
The /backtests/orders/read API provides a response in the following format:
200 Success
BacktestOrdersResponse Model - Contains orders and the number of orders of the backtest in the request criteria.
orders
Order object
/.
length
integer
Total number of returned orders.
{
E
x
a
m
ple
"
o
r
d
e
r
s
": {
"
i
d
": 0
,
"
c
o
n
t
i
n
g
e
n
t
I
d
": 0
,
"
b
r
o
k
e
r
I
d
": [
"
s
t
r
i
n
g
"
]
,
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
l
i
m
i
t
P
r
i
c
e
": ,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
s
t
o
p
T
r
i
g
g
e
r
e
d
": ,
"
p
r
i
c
e
": 0
,
"
p
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
t
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
F
i
l
l
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
U
p
d
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
a
n
c
e
l
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
y
p
e
": 0
=
M
a
r
k
e
t
,
"
s
t
a
t
u
s
": 0
=
N
e
w
,
"
t
a
g
": "
s
t
r
i
n
g
"
,
"
s
e
c
u
r
i
t
y
T
y
p
e
": 0
=
B
a
s
e
,
"
d
i
r
e
c
t
i
o
n
": 0
=
B
u
y
,
"
v
a
l
u
e
": 0
,
"
o
r
d
e
r
S
u
b
m
i
s
s
i
o
n
D
a
t
a
": {
"
b
i
d
P
r
i
c
e
": 0
,
"
a
s
k
P
r
i
c
e
": 0
,
"
l
a
s
t
P
r
i
c
e
": 0
}
,
"
i
s
M
a
r
k
e
t
a
b
l
e
": t
r
u
e
,
"
p
r
o
p
e
r
t
i
e
s
": {
"
t
i
m
e
I
n
F
o
r
c
e
": 0
=
G
o
o
d
T
i
l
C
a
n
c
e
l
e
d
}
,
"
e
v
e
n
t
s
": [
{
"
a
l
g
o
r
i
t
h
m
I
d
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
V
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
P
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
,
"
o
r
d
e
r
I
d
": 0
,
"
o
r
d
e
r
E
v
e
n
t
I
d
": 0
,
"
i
d
": 0
,
"
s
t
a
t
u
s
": "
n
e
w
"
,
"
o
r
d
e
r
F
e
e
A
m
o
u
n
t
": 0
,
"
o
r
d
e
r
F
e
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
P
r
i
c
e
": 0
,
"
f
i
l
l
P
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
Q
u
a
n
t
i
t
y
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
e
s
s
a
g
e
": "
s
t
r
i
n
g
"
,
"
i
s
A
s
s
i
g
n
m
e
n
t
": t
r
u
e
,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
i
m
e
": 0
,
"
i
s
I
n
T
h
e
M
o
n
e
y
": }
]
,
"
t
r
a
i
l
i
n
g
A
m
o
u
n
t
": 0
,
"
t
r
a
i
l
i
n
g
P
e
r
c
e
n
t
a
g
e
": ,
"
g
r
o
u
p
O
r
d
e
r
M
a
n
a
g
e
r
": {
"
i
d
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
c
o
u
n
t
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
o
r
d
e
r
I
d
s
": [
"
i
n
t
e
g
e
r
"
]
,
"
d
i
r
e
c
t
i
o
n
": 0
}
,
"
t
r
i
g
g
e
r
P
r
i
c
e
": 0
,
"
t
r
i
g
g
e
r
T
o
u
c
h
e
d
": },"length": 0
}
Order Model - Order struct for placing new trade.
id
integer
Order ID.
contingentId
integer
Order Id to process before processing this order.
brokerId
string Array
Brokerage Id for this order for when the brokerage splits
orders into multiple pieces.
symbol
Symbol object
Represents a unique security identifier. This is made of two
components, the unique SID and the Value. The value is the
current ticker symbol while the SID is constant over the life
of a security.
limitPrice
nummber
Limit price of the Order.
stopPrice
number
Stop price of the Order.
stopTriggered
bool
Indicates if the stop price has been reached, so the limit
order has been triggered.
price
number
Price of the Order.
priceCurrency
string
Currency for the order price.
time
string($date-time)
Gets the utc time the order was created.
createdTime
string($date-time)
Gets the utc time this order was created. Alias for Time.
lastFillTime
string($date-time)
Gets the utc time the last fill was received, or null if no fills
have been received.
lastUpdateTime
string($date-time)
Gets the utc time this order was last updated, or null if the
order has not been updated.
canceledTime
string($date-time)
Gets the utc time this order was canceled, or null if the
order was not canceled.
quantity
number
Number of shares to execute.
type
integer Enum
Order type. Options : ['0 = Market', '1 = Limit', '2 =
StopMarket', '3 = StopLimit', '4 = MarketOnOpen', '5 = MarketOnClose', '6 = OptionExercise', '7 = LimitIfTouched',
'8 = ComboMarket', '9 = ComboLimit', '10 =
ComboLegLimit', '11 = TrailingStop']
status
integer Enum
Status of the Order. Options : ['0 = New', '1 = Submitted', '2
= PartiallyFilled', '3 = Filled', '5 = Canceled', '6 = None', '7 =
Invalid', '8 = CancelPending', '9 = UpdateSubmitted']
tag
string
Tag the order with some custom data.
securityType
integer Enum
Type of tradable security / underlying asset. Options : ['0 =
Base', '1 = Equity', '2 = Option', '3 = Commodity', '4 =
Forex', '5 = Future', '6 = Cfd', '7 = Crypto']
direction
integer Enum
Order Direction Property based off Quantity. Options : ['0 =
Buy', '1 = Sell', '2 = Hold']
value
number
Gets the executed value of this order. If the order has not
yet filled, then this will return zero.
orderSubmissionData
OrderSubmissionData object
Stores time and price information available at the time an
order was submitted.
isMarketable
boolean
Returns true if the order is a marketable order.
properties
OrderProperties object
Additional properties of the order.
events
OrderEvent Array
The order events.
trailingAmount
number
Trailing amount for a trailing stop order.
trailingPercentage
bool
Determines whether the trailingAmount is a percentage or
an absolute currency value.
groupOrderManager
GroupOrderManager object
Manager of a group of orders.
triggerPrice
number
The price which, when touched, will trigger the setting of a
limit order at limitPrice.
triggerTouched
bool
Whether or not the triggerPrice has been touched.
E
x
a
m
ple
{
"
i
d
": 0
,
"
c
o
n
t
i
n
g
e
n
t
I
d
": 0
,
"
b
r
o
k
e
r
I
d
": [
"
s
t
r
i
n
g
"
]
,
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
l
i
m
i
t
P
r
i
c
e
": ,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
s
t
o
p
T
r
i
g
g
e
r
e
d
": ,
"
p
r
i
c
e
": 0
,
"
p
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
t
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
F
i
l
l
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
U
p
d
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
a
n
c
e
l
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
y
p
e
": 0
=
M
a
r
k
e
t
,
"
s
t
a
t
u
s
": 0
=
N
e
w
,
"
t
a
g
": "
s
t
r
i
n
g
"
,
"
s
e
c
u
r
i
t
y
T
y
p
e
": 0
=
B
a
s
e
,
"
d
i
r
e
c
t
i
o
n
": 0
=
B
u
y
,
"
v
a
l
u
e
": 0
,
"
o
r
d
e
r
S
u
b
m
i
s
s
i
o
n
D
a
t
a
": {
"
b
i
d
P
r
i
c
e
": 0
,
"
a
s
k
P
r
i
c
e
": 0
,
"
l
a
s
t
P
r
i
c
e
": 0
}
,
"
i
s
M
a
r
k
e
t
a
b
l
e
": t
r
u
e
,
"
p
r
o
p
e
r
t
i
e
s
": {
"
t
i
m
e
I
n
F
o
r
c
e
": 0
=
G
o
o
d
T
i
l
C
a
n
c
e
l
e
d
}
,
"
e
v
e
n
t
s
": [
{
"
a
l
g
o
r
i
t
h
m
I
d
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
V
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
P
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
,
"
o
r
d
e
r
I
d
": 0
,
"
o
r
d
e
r
E
v
e
n
t
I
d
": 0
,
"
i
d
": 0
,
"
s
t
a
t
u
s
": "
n
e
w
"
,
"
o
r
d
e
r
F
e
e
A
m
o
u
n
t
": 0
,
"
o
r
d
e
r
F
e
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
P
r
i
c
e
": 0
,
"
f
i
l
l
P
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
Q
u
a
n
t
i
t
y
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
e
s
s
a
g
e
": "
s
t
r
i
n
g
"
,
"
i
s
A
s
s
i
g
n
m
e
n
t
": t
r
u
e
,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
i
m
e
": 0
,
"
i
s
I
n
T
h
e
M
o
n
e
y
": }
]
,
"
t
r
a
i
l
i
n
g
A
m
o
u
n
t
": 0
,
"
t
r
a
i
l
i
n
g
P
e
r
c
e
n
t
a
g
e
": ,
"
g
r
o
u
p
O
r
d
e
r
M
a
n
a
g
e
r
": {
"
i
d
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
c
o
u
n
t
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
o
r
d
e
r
I
d
s
": [
"
i
n
t
e
g
e
r
"
]
,
"
d
i
r
e
c
t
i
o
n
": 0
}
,
"
t
r
i
g
g
e
r
P
r
i
c
e
": 0
,
"
t
r
i
g
g
e
r
T
o
u
c
h
e
d
": }
Symbol Model - Represents a unique security identifier. This is made of two components, the unique SID and the Value. The
value is the current ticker symbol while the SID is constant over the life of a security.
value
string
The current symbol for this ticker.
id
string
The security identifier for this symbol.
permtick
string
The current symbol for this ticker.
Example
{
"value": "string",
"id": "string",
"permtick": "string"
}
OrderSubmissionData Model - Stores time and price information available at the time an order was submitted.
bidPrice
number
The bid price at an order submission time.
askPrice
number
The ask price at an order submission time.
lastPrice
number
The current price at an order submission time.
Example
{
"bidPrice": 0,
"askPrice": 0,
"lastPrice": 0
}
OrderProperties Model - Additional properties of the order.
timeInForce
object Enum
Defines the length of time over which an order will continue
working before it is cancelled. Options : ['0 =
GoodTilCanceled', '1 = Day', '2 = GoodTilDate']
Example
{
"timeInForce": 0 = GoodTilCanceled
}
OrderEvent Model - Change in an order state applied to user algorithm portfolio
algorithmId
string
Algorithm Id, BacktestId or DeployId.
symbol
string
Easy access to the order symbol associated with this event.
symbolValue
string
The current symbol for this ticker; It is a user friendly
symbol representation.
symbolPermtick
string
The original symbol used to generate this symbol.
orderId
integer
Id of the order this event comes from.
orderEventId
integer
The unique order event id for each order.
id
integer
The unique order event Id for each order.
status
string Enum
Status of the Order. Options : ['new', 'submitted',
'partiallyFilled', 'filled', 'canceled', 'none', 'invalid',
'cancelPending', 'updateSubmitted']
orderFeeAmount
number
The fee amount associated with the order.
orderFeeCurrency
string
The fee currency associated with the order.
fillPrice
number
Fill price information about the order.
fillPriceCurrency
string
Currency for the fill price.
fillQuantity
number
Number of shares of the order that was filled in this event.
direction
string
Order direction.
message
string
Any message from the exchange.
isAssignment
boolean
True if the order event is an assignment.
stopPrice
number
The current stop price.
limitPrice
number
The current limit price.
q
u
a
n
tit
y
n
u
m
b
e
r
T
h
e
c
u
r
r
e
n
t
o
r
d
e
r
q
u
a
n
tit
y. time integer The time of this event in unix
tim
e
s
t
a
m
p. isInTheMoney bool True if the order event's option is In-The
-
M
o
n
e
y
(IT
M
). Example { "algorithmId": "string", "symbol": "string", "symbolValue": "string", "symbolPermtick": "string", "orderId": 0, "orderEventId": 0, "id": 0, "status": "new", "orderFeeAmount": 0, "orderFeeCurrency": "string", "fillPrice": 0, "fillPriceCurrency": "string", "fillQuantity": 0, "direction": "string", "message": "string", "isAssignment": true, "stopPrice": 0, "limitPrice": 0, "quantity": 0, "time": 0, "isInTheMoney": }
GroupOrderManager Model - Manager of a group of orders.
id
integer
The unique order group Id.
quantity
number
The group order quantity.
count
integer
The total order count associated with this order group.
limitPrice
number
The limit price associated with this order group if any.
orderIds
integer Array
The order Ids in this group.
direction
integer
Order Direction Property based off Quantity.
Example
{
"id": 0,
"quantity": 0,
"count": 0,
"limitPrice": 0,
"orderIds": [
"integer"
],
"direction": 0
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > Read Backtest > Insights
Read Backtest
Insights
Introduction
Read out the insights of a backtest.
Request
Fetch the insights of a backtest for the project Id, backtest Id and steps provided. The /backtests/read/insights API accepts
requests in the following format:
ReadBacktestInsightsRequest Model - Request to read insights from a backtest.
start
integer
Starting index of the insights to be fetched. Required if end
> 100.
end
integer
Last index of the insights to be fetched. Note that end -
start must be less than 100.
projectId
integer
Id of the project from which to read the backtest.
backtestId
string
Id of the backtest from which to read the insights.
Example
{
"start": 0,
"end": 0,
"projectId": 0,
"backtestId": "string"
}
Responses
The /backtests/read/insights API provides a response in the following format:
200 Success
BacktestInsightsResponse Model - Contains insights and the number of insights of the backtest in the request criteria.
insights
Insight Array
Collection of insights.
length
integer
Total number of returned insights.
success
boolean
Indicate if the API request was successful.
Example
{
"insights": [
{
"id": "string",
"groupId": "string",
"sourceModel": "string",
"generatedTime": "string",
"createdTime": 0,
"closeTime": 0,
"symbol": "string",
"ticker": "string",
"type": "price",
"reference": "string",
"referenceValueFinal": "string",
"direction": "down",
"period": 0,
"magnitude": 0,
"confidence": 0,
"weight": 0,
"scoreIsFinal": ,
"scoreDirection": 0,
"scoreMagnitude": 0,
"estimatedValue": 0,
"tag": "2021-11-26T15:18:27.693Z"
}
],
"length": 0,
"success": true
}
Insight Model - Insight struct for emitting new prediction.
id
string
Insight ID.
groupId
string
ID of the group of insights.
sourceModel
string
Name of the model that sourced the insight.
generatedTime
string
Gets the utc unixtime this insight was generated.
createdTime
number
Gets the utc unixtime this insight was created.
closeTime
number
Gets the utc unixtime this insight was closed.
symbol
string
Gets the symbol ID this insight is for.
ticker
string
Gets the symbol ticker this insight is for.
type
string Enum
Gets the type of insight, for example, price or volatility.
Options : ['price', 'volatility']
reference
string
Gets the initial reference value this insight is predicting
against.
referenceValueFinal
string
Gets the final reference value, used for scoring, this insight
is predicting against.
direction
string Enum
Gets the predicted direction, down, flat or up. Options :
['down', 'flat', 'up']
period
number
Gets the period, in seconds, over which this insight is
expected to come to fruition.
magnitude
number
Gets the predicted percent change in the insight type
(price/volatility). This value can be null.
confidence
number
Gets the confidence in this insight. This value can be null.
weight
number
Gets the portfolio weight of this insight. This value can be
null.
scoreIsFinal
bool
Gets whether or not this is the insight's final score.
scoreDirection
number
Gets the direction score.
scoreMagnitude
number
Gets the magnitude score.
estimatedValue
number
Gets the estimated value of this insight in the account
currency.
tag
string($float)
The insight's tag containing additional information.
Example
{
"id": "string",
"groupId": "string",
"sourceModel": "string",
"generatedTime": "string",
"createdTime": 0,
"closeTime": 0,
"symbol": "string",
"ticker": "string",
"type": "price",
"reference": "string",
"referenceValueFinal": "string",
"direction": "down",
"period": 0,
"magnitude": 0,
"confidence": 0,
"weight": 0,
"scoreIsFinal": ,
"scoreDirection": 0,
"scoreMagnitude": 0,
"estimatedValue": 0,
"tag": "2021-11-26T15:18:27.693Z"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > Update Backtest
Backtest Management
Update Backtest
Introduction
Update a backtest name or note.
Request
A JSON object containing info about the backtest and new name. The /backtests/update API accepts requests in the following
format:
UpdateBacktestRequest Model - Request to update a backtest's name.
projectId
integer
Project Id for the backtest we want to update.
backtestId
string
Backtest Id we want to update.
name
string
Name we would like to assign to the backtest.
note
string
Note attached to the backtest.
Example
{
"projectId": 0,
"backtestId": "string",
"name": "string",
"note": "string"
}
Responses
The /backtests/update API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > Delete Backtest
Backtest Management
Delete Backtest
Introduction
Delete a backtest from the specified project and backtestId.
Request
Information required to delete the backtest. The /backtests/delete API accepts requests in the following format:
DeleteBacktestRequest Model - Request to delete a backtest.
projectId
integer
Project Id for the backtest we want to delete.
backtestId
string
Backtest Id we want to delete.
Example
{
"projectId": 0,
"backtestId": "string"
}
Responses
The /backtests/delete API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Backtest Management > List Backtests
Backtest Management
List Backtests
Introduction
List all the backtests for the project.
Request
Fetch the results for the project Id provided. The /backtests/list API accepts requests in the following format:
ListBacktestRequest Model - Request to list the backtests from a project.
projectId
integer
Id of the project from which to read one or multiple
backtests.
includeStatistics
boolean
If true, the backtests summaries from the response will
contain the statistics with their corresponding values.
Example
{
"projectId": 0,
"includeStatistics": true
}
Responses
The /backtests/list API provides a response in the following format:
200 Success
BacktestSummaryResponse Model - Collection container for a list of backtest summaries for a project.
backtest
BacktestSummaryResult Array
Collection of backtest summaries for a project.
count
int
Number of backtest summaries retrieved in the response.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"backtest": [
{
"backtestId": "string",
"status": "Completed.",
"name": "string",
"created": "2021-11-26T15:18:27.693Z",
"progress": 0,
"optimizationId": "string",
"tradeableDates": 0,
"parameterSet": {
"name": "string",
"value": 0
},
"snapshotId": 0,
"tags": [
"string"
],
"sharpeRatio": ,
"alpha": ,
"beta": ,
"compoundingAnnualReturn": ,
"drawdown": ,
"lossRate": ,
"netProfit": ,
"parameters": ,
"psr": ,
"securityTypes": "string",
"sortinoRatio": ,
"trades": ,
"treynorRatio": ,
"winRate":
}
],
"count": ,
"success": true,
"errors": [
"string"
]
}
BacktestSummaryResult Model - Result object class for the List Backtest response from the API.
backtestId
string
Assigned backtest ID.
status
string Enum
Status of the backtest. Options : ['Completed.', 'In
Queue...', "'Running: _%'"]
name
string
Name of the backtest.
created
string($date-time)
Backtest creation date and time.
progress
number
Progress of the backtest in percent 0-1.
optimizationId
string
Optimization task ID, if the backtest is part of an
optimization.
tradeableDates
integer
Number of traadeable days.
parameterSet
ParameterSet object
Parameters used in the backtest.
snapshotId
integer
Snapshot id of this backtest result.
tags
string Array
Collection of tags for the backtest.
sharpeRatio
float
Sharpe ratio with respect to risk free rate; measures excess
of return per unit of risk.
alpha
float
Algorithm "Alpha" statistic - abnormal returns over the risk
free rate and the relationshio (beta) with the benchmark
returns.
beta
float
Algorithm "beta" statistic - the covariance between the
algorithm and benchmark performance, divided by
benchmark's variance.
compoundingAnnualReturn
float
Annual compounded returns statistic based on the finalstarting capital and years.
drawdown
float
Drawdown maximum percentage.
lossRate
float
The ratio of the number of losing trades to the total number
of trades.
netProfit
float
Net profit percentage.
parameters
int
Number of parameters in the backtest.
psr
float
Price-to-sales ratio.
securityTypes
string
SecurityTypes present in the backtest.
sortinoRatio
float
Sortino ratio with respect to risk free rate; measures excess
of return per unit of downside risk.
trades
int
Number of trades in the backtest.
treynorRatio
float
Treynor ratio statistic is a measurement of the returns
earned in excess of that which could have been earned on
an investment that has no diversifiable risk.
winRate
float
The ratio of the number of winning trades to the total
number of trades.
E
x
a
m
ple
{
"
b
a
c
k
t
e
s
t
I
d
": "
s
t
r
i
n
g
"
,
"
s
t
a
t
u
s
": "
C
o
m
p
l
e
t
e
d."
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
p
r
o
g
r
e
s
s
": 0
,
"
o
p
t
i
m
i
z
a
t
i
o
n
I
d
": "
s
t
r
i
n
g
"
,
"
t
r
a
d
e
a
b
l
e
D
a
t
e
s
": 0
,
"
p
a
r
a
m
e
t
e
r
S
e
t
": {
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
v
a
l
u
e
": 0
}
,
"
s
n
a
p
s
h
o
t
I
d
": 0
,
"
t
a
g
s
": [
"
s
t
r
i
n
g
"
]
,
"
s
h
a
r
p
e
R
a
t
i
o
": ,
"
a
l
p
h
a
": ,
"
b
e
t
a
": ,
"
c
o
m
p
o
u
n
d
i
n
g
A
n
n
u
a
l
R
e
t
u
r
n
": ,
"
d
r
a
w
d
o
w
n
": ,
"
l
o
s
s
R
a
t
e
": ,
"
n
e
t
P
r
o
f
i
t
": ,
"
p
a
r
a
m
e
t
e
r
s
": ,
"
p
s
r
": ,
"
s
e
c
u
r
i
t
y
T
y
p
e
s
": "
s
t
r
i
n
g
"
,
"
s
o
r
t
i
n
o
R
a
t
i
o
": ,
"
t
r
a
d
e
s
": ,
"
t
r
e
y
n
o
r
R
a
t
i
o
": ,
"
w
i
n
R
a
t
e
": }
ParameterSet Model - Parameter set.
name
string
Name of parameter.
value
number
Value of parameter.
Example
{
"name": "string",
"value": 0
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management
API Reference
Live Management
The QuantConnect REST API lets you manage your live algorithms on our cloud servers through URL endpoints.
Create Live Algorithm
Read Live Algorithm
Update Live Algorithm
List Live Algorithms
Live Commands
API Reference > Live Management > Create Live Algorithm
Live Management
Create Live Algorithm
Introduction
Create a live algorithm.
Request
Project, compile and brokerage login information for deploying a live algorithm. The /live/create API accepts requests in the
following format:
CreateLiveAlgorithmRequest Model - Request to create a live algorithm.
versionId
string
example: -1
The version of the Lean used to run the algorithm. -1 is
master, however, sometimes this can create problems with
live deployments. If you experience problems using, try
specifying the version of Lean you would like to use.
projectId
integer
Project Id.
compileId
string
Compile Id.
nodeId
string
Id of the node that will run the algorithm.
brokerage
object Enum
Brokerage configurations to be used in the live algorithm.
Options : ['QuantConnectSettings',
'InteractiveBrokersSettings', 'BinanceSettings',
'BinanceFuturesUSDMSettings',
'BinanceFuturesCOINSettings', 'BinanceUSSettings',
'TradierSettings', 'BitfinexSettings', 'CoinbaseSettings',
'KrakenSettings', 'BybitSettings', 'OandaSettings',
'ZerodhaSettings', 'SamcoSettings', 'WolverineSettings',
'CharlesSchwabSettings', 'TradingTechnologiesSettings',
'RBIBrokerageSettings', 'TerminalLinkSettings']
dataProviders
object
Dictionary of data provider configurations to be used in the
live algorithm.
E
x
a
m
ple
{
"
v
e
r
s
i
o
n
I
d
": "
-
1
"
,
"
p
r
o
j
e
c
t
I
d
": 0
,
"
c
o
m
p
i
l
e
I
d
": "
s
t
r
i
n
g
"
,
"
n
o
d
e
I
d
": "
s
t
r
i
n
g
"
,
"
b
r
o
k
e
r
a
g
e
": {
"
i
d
": "
s
t
r
i
n
g
"
,
"
h
o
l
d
i
n
g
s
": [
{
"
s
y
m
b
o
l
I
d
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
": "
s
t
r
i
n
g
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
a
v
e
r
a
g
e
P
r
i
c
e
": 0
}
]
,
"
c
a
s
h
": [
{
"
a
m
o
u
n
t
": 0
,
"
c
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
}
]
}
,
"
d
a
t
a
P
r
o
v
i
d
e
r
s
": {
"
Q
u
a
n
t
C
o
n
n
e
c
t
B
r
o
k
e
r
a
g
e
": {
"
i
d
": "
s
t
r
i
n
g
"
,
"
h
o
l
d
i
n
g
s
": [
{
"
s
y
m
b
o
l
I
d
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
": "
s
t
r
i
n
g
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
a
v
e
r
a
g
e
P
r
i
c
e
": 0
}
]
,
"
c
a
s
h
": [
{
"
a
m
o
u
n
t
": 0
,
"
c
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
}
]
}
}
}
QuantConnectSettings Model - QuantConnect settings for using it as a brokerage or data provider.
id
string
ID of QuantConnect, this is QuantConnectBrokerage.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
cash
CashAmount Array
List of cash amount.
Example
{
"id": "string",
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
],
"cash": [
{
"amount": 0,
"currency": "string"
}
]
}
InteractiveBrokersSettings Model - Settings for using Interactive Brokers as brokerage or data provider.
id
string
ID of InteractiveBrokers, this is
InteractiveBrokersBrokerage.
ib-user-name
string
Your Interactive Brokers username.
ib-password
string
Your Interactive Brokers password.
ib-trading-mode
string Enum
Represents the types of environments supported by
Interactive Brokers for trading. Options : ['live', 'paper']
ib-account
string
Your Interactive Brokers account id.
ib-weekly-restart-utc-time
string($date)
Weekly restart UTC time (hh:mm:ss).
Example
{
"id": "string",
"ib-user-name": "string",
"ib-password": "string",
"ib-trading-mode": "live",
"ib-account": "string",
"ib-weekly-restart-utc-time": "2021-11-
26T15:18:27.693Z"
}
BinanceSettings Model
id
string
ID of the brokerage, this is, BinanceBrokerage.
binance-exchange-name
string
Binance exchange, this is, Binance.
binance-api-secret
string
Your Binance API secret.
binance-api-key
string
Your Binance API key.
binance-api-url
string
Binance configuration for spot/margin. The value for this
property is https://api.binance.com.
binance-websocket-url
string
Binance configuration for spot/margin. The value for this
property is wss://stream.binance.com:9443/ws.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
description
object
/.
Example
{
"id": "string",
"binance-exchange-name": "string",
"binance-api-secret": "string",
"binance-api-key": "string",
"binance-api-url": "string",
"binance-websocket-url": "string",
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
],
"description":
}
BinanceFuturesUSDMSettings Model
id
string
ID of the brokerage, this is, BinanceBrokerage.
binance-exchange-name
string
Binance exchange, this is, Binance-USDM-Futures.
binance-api-secret
string
Your Binance API secret.
binance-api-key
string
Your Binance API key.
binance-fapi-url
string
Binance Futures configuration for spot/margin. The value
for this property is https://fapi.binance.com.
binance-fwebsocket-url
string
Binance Futures configuration for spot/margin. The value
for this property is wss://fstream.binance.com/ws.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
description
object
/.
Example
{
"id": "string",
"binance-exchange-name": "string",
"binance-api-secret": "string",
"binance-api-key": "string",
"binance-fapi-url": "string",
"binance-fwebsocket-url": "string",
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
],
"description":
}
BinanceFuturesCOINSettings Model
id
string
ID of the brokerage, this is, BinanceBrokerage.
binance-exchange-name
string
Binance exchange, this is, Binance-COIN-Futures.
binance-api-secret
string
Your Binance API secret.
binance-api-key
string
Your Binance API key.
binance-dapi-url
string
Binance Futures configuration for spot/margin. The value
for this property is https://dapi.binance.com.
binance-dwebsocket-url
string
Binance Futures configuration for spot/margin. The value
for this property is wss://dstream.binance.com/ws.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
description
object
/.
Example
{
"id": "string",
"binance-exchange-name": "string",
"binance-api-secret": "string",
"binance-api-key": "string",
"binance-dapi-url": "string",
"binance-dwebsocket-url": "string",
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
],
"description":
}
BinanceUSSettings Model
id
string
ID of the brokerage, this is, BinanceBrokerage.
binance-exchange-name
string
Binance exchange, this is, BinanceUS.
binanceus-api-secret
string
Your Binance US API secret.
binanceus-api-key
string
Your Binance US API key.
binanceus-api-url
string
Binance US configuration for spot/margin. The value for this
property is https://api.binance.us.
binanceus-websocket-url
string
Binance US configuration for spot/margin. The value for this
property is wss://stream.binance.us:9443/ws.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
description
object
/.
Example
{
"id": "string",
"binance-exchange-name": "string",
"binanceus-api-secret": "string",
"binanceus-api-key": "string",
"binanceus-api-url": "string",
"binanceus-websocket-url": "string",
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
],
"description":
}
TradierSettings Model - Settings for using Tradier as a data provider.
id
string
ID of the brokerage, this is, TradierBrokerage.
tradier-account-id
string
Your Tradier account id.
tradier-access-token
string
Your Tradier access token.
tradier-environment
string Enum
Whether the developer sandbox should be used. Options :
['live', 'paper']
Example
{
"id": "string",
"tradier-account-id": "string",
"tradier-access-token": "string",
"tradier-environment": "live"
}
BitfinexSettings Model - Settings for using Bitfinex as a data provider or brokerage.
id
string
ID of the brokerage, this is, BitfinexBrokerage.
bitfinex-api-key
string
Your Bitfinex API key.
bitfinex-api-secret
string
Your Bitfinex API secret.
Example
{
"id": "string",
"bitfinex-api-key": "string",
"bitfinex-api-secret": "string"
}
CoinbaseSettings Model - Settings for using Coinbase as a data provider or brokerage.
id
string
ID of the brokerage, this is, CoinbaseBrokerage.
coinbase-api-key
string
Your Coinbase Advanced Trade API key.
coinbase-api-secret
string
Your Coinbase Advanced Trade API secret.
coinbase-url
string
Coinbase URL, this is, wss://advanced-tradews.coinbase.com.
coinbase-rest-api
string
Coinbase REST API, this is, https://api.coinbase.com.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
Example
{
"id": "string",
"coinbase-api-key": "string",
"coinbase-api-secret": "string",
"coinbase-url": "string",
"coinbase-rest-api": "string",
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
]
}
KrakenSettings Model - Settings for using Kraken as a data provider.
id
string
ID of the brokerage, this is, KrakenBrokerage.
kraken-api-key
string
Your Kraken API key.
kraken-api-secret
string
Your Kraken API secret.
kraken-verification-tier
string
Your Kraken Verification Tier.
Example
{
"id": "string",
"kraken-api-key": "string",
"kraken-api-secret": "string",
"kraken-verification-tier": "string"
}
BybitSettings Model - Settings for using Bybit as a data provider or brokerage.
id
string
ID of the brokerage, this is, BybitBrokerage.
bybit-api-key
string
Your Bybit API key.
bybit-api-secret
string
Your Bybit API secret.
bybit-vip-level
string
Your Bybit VIP Level.
bybit-use-testnet
string Enum
Whether the testnet should be used. Options : ['live',
'paper']
bybit-api-url
string
Bybit API URL, this is, https://api-testnet.bybit.com.
bybit-websocket-url
string
Bybit Websocket URL, this is, wss://stream.bybit.com.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
Example
{
"id": "string",
"bybit-api-key": "string",
"bybit-api-secret": "string",
"bybit-vip-level": "string",
"bybit-use-testnet": "live",
"bybit-api-url": "string",
"bybit-websocket-url": "string",
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
]
}
OandaSettings Model - Settings for using Oanda as a data provider or brokerage.
id
string
ID of the brokerage, this is, OandaBrokerage.
oanda-account-id
string
Your OANDA account id can be found on your OANDA
Account Statement page
(https://www.oanda.com/account/statement/). It follows
the following format '###-###-######-###'.
oanda-access-token
string
Your OANDA API token. You can generate an API token
from the Manage API Access page
(https://www.oanda.com/account/tpa/personal_token).
oanda-environment
string Enum
The environment to run in, Practice for fxTrade Practice,
Trade for fxTrade. Options : ['Practice', 'Trade']
Example
{
"id": "string",
"oanda-account-id": "string",
"oanda-access-token": "string",
"oanda-environment": "Practice"
}
ZerodhaSettings Model - Settings for using Zerodha as a data provider or brokerage.
id
string
Brokerage ID, this is, ZerodhaBrokerage.
zerodha-api-key
string
Your Kite Connect API key.
zerodha-access-token
string
Your Kite Connect access token.
zerodha-product-type
string Enum
The product type must be set to MIS if you are targeting
intraday products, CNC if you are targeting delivery
products or NRML if you are targeting carry forward
products. Options : ['mis', 'cnc', 'nrml']
zerodha-trading-segment
string Enum
The trading segment must be set to 'equity' if you are
trading equities on NSE or BSE, or 'commodity' if you are
trading commodities on MCX. Options : ['equity',
'commodity']
zerodha-history-subscription
bool
Whether you have a history API subscription for Zerodha.
Example
{
"id": "string",
"zerodha-api-key": "string",
"zerodha-access-token": "string",
"zerodha-product-type": "mis",
"zerodha-trading-segment": "equity",
"zerodha-history-subscription":
}
SamcoSettings Model - Settings for using Samco as a data provider or brokerage.
id
string
Brokerage ID, this is, SamcoBrokerage.
samco-client-id
string
Your Samco account Client ID.
samco-client-password
string
Your Samco account password.
samco-year-of-birth
int
Your year of birth (YYYY) registered with Samco.
samco-product-type
string Enum
MIS if you are targeting intraday products, CNC if you are
targeting delivery products, NRML if you are targeting carry
forward products. Options : ['mis', 'cnc', 'nrml']
samco-trading-segment
string Enum
\'equity\' if you are trading equities on NSE or BSE,
commodity if you are trading \'commodities\' on MCX.
Options : ['equity', 'commodity']
Example
{
"id": "string",
"samco-client-id": "string",
"samco-client-password": "string",
"samco-year-of-birth": ,
"samco-product-type": "mis",
"samco-trading-segment": "equity"
}
WolverineSettings Model - Settings for using Wolverine Execution Services as a brokerage.
id
string
Brokerage ID, this is, WolverineBrokerage.
wolverine-on-behalf-of-comp-id
string
Value used to identify the trading firm.
wolverine-account
string
Wolverine Execution Services account name.
cash
CashAmount Array
List of cash amount.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
Example
{
"id": "string",
"wolverine-on-behalf-of-comp-id": "string",
"wolverine-account": "string",
"cash": [
{
"amount": 0,
"currency": "string"
}
],
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
]
}
CharlesSchwabSettings Model - Settings for using Charles Schwab as a data provider or brokerage.
id
string
Brokerage ID, this is, Charles Schwab.
charles-schwab-app-key
string
Your Charles Schwab app key.
charles-schwab-secret
string
Your Charles Schwab secret.
charles-schwab-account-number
string
Your Charles Schwab account number.
Example
{
"id": "string",
"charles-schwab-app-key": "string",
"charles-schwab-secret": "string",
"charles-schwab-account-number": "string"
}
TradingTechnologiesSettings Model - Settings for using Trading Technologies as a brokerage.
id
string
Brokerage ID, this is, TradingTechnologiesBrokerage.
tt-user-name
string
Trading Technologies user name.
tt-session-password
string
Trading Technologies session password.
tt-account-name
string
Trading Technologies account name.
tt-rest-app-key
string
Trading Technologies App key.
tt-rest-app-secret
string
Trading Technologies App secret.
tt-rest-environment
string Enum
Environment in which the brokerage Trading Technologies
will be used. Options : ['live', 'uat']
tt-order-routing-sender-comp-id
string
Trading Technologies remote comp id.
cash
CashAmount Array
List of cash amount.
Example
{
"id": "string",
"tt-user-name": "string",
"tt-session-password": "string",
"tt-account-name": "string",
"tt-rest-app-key": "string",
"tt-rest-app-secret": "string",
"tt-rest-environment": "live",
"tt-order-routing-sender-comp-id": "string",
"cash": [
{
"amount": 0,
"currency": "string"
}
]
}
RBIBrokerageSettings Model - Settings for using RBI as a brokerage.
id
string
Brokerage ID, this is, RBIBrokerage.
rbi-on-behalf-of-comp-id
string
Value used to identify the trading firm.
rbi-account
string
RBI account name.
cash
CashAmount Array
List of cash amount.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
Example
{
"id": "string",
"rbi-on-behalf-of-comp-id": "string",
"rbi-account": "string",
"cash": [
{
"amount": 0,
"currency": "string"
}
],
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
]
}
TerminalLinkSettings Model - Settings for using TerminalLink as a brokerage.
id
string
Brokerage ID, this is TerminalLinkBrokerage.
terminal-link-connection-type
string Enum
Terminal Link Connection Type [DAPI, SAPI]. Options :
['DAPI', 'SAPI']
cash
CashAmount Array
List of cash amount.
holdings
BrokerageHolding Array
List of holdings for the brokerage.
Example
{
"id": "string",
"terminal-link-connection-type": "DAPI",
"cash": [
{
"amount": 0,
"currency": "string"
}
],
"holdings": [
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
]
}
BrokerageHolding Model - Holding object class for creating a live algorithm.
symbolId
string
Symbol ID of the holding.
symbol
string
Symbol ticker of the holding.
quantity
number
Quantity of the Symbol we hold.
averagePrice
number
Average Price of our Holding in the currency the symbol is
traded in.
Example
{
"symbolId": "string",
"symbol": "string",
"quantity": 0,
"averagePrice": 0
}
CashAmount Model - Represents a cash amount which can be converted to account currency using a currency converter.
amount
number
The amount of cash.
currency
string
The currency in which the cash amount is denominated.
Example
{
"amount": 0,
"currency": "string"
}
Responses
The /live/create API provides a response in the following format:
200 Success
CreateLiveAlgorithmResponse Model - Response received when deploying a live algorithm.
live
LiveAlgorithm object
Live algorithm instance result from the QuantConnect Rest
API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"live": {
"projectId": 0,
"deployId": "string",
"status": "DeployError",
"launched": "2021-11-26T15:18:27.693Z",
"stopped": "2021-11-26T15:18:27.693Z",
"brokerage": "Interactive",
"subscription": "string",
"error": "string",
"success": true,
"errors": [
"string"
]
},
"success": true,
"errors": [
"string"
]
}
LiveAlgorithm Model - Live algorithm instance result from the QuantConnect Rest API.
projectId
integer
Project Id for the live instance.
deployId
string
Unique live algorithm deployment identifier (similar to a
backtest id).
status
string Enum
States of a live deployment. Options : ['DeployError',
'InQueue', 'Running', 'Stopped', 'Liquidated', 'Deleted',
'Completed', 'RuntimeError', 'Invalid', 'LoggingIn',
'Initializing', 'History']
launched
string($date-time)
Datetime the algorithm was launched in UTC.
stopped
string($date-time)
Datetime the algorithm was stopped in UTC, null if its still
running.
brokerage
string Enum
Brokerage. Options : ['Interactive', 'FXCM', 'Oanda',
'Tradier', 'PaperBrokerage', 'Alpaca', 'Bitfinex', 'Binance',
'Coinbase']
subscription
string
Chart we're subscribed to.
error
string
Live algorithm error message from a crash or algorithm
runtime error.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"projectId": 0,
"deployId": "string",
"status": "DeployError",
"launched": "2021-11-26T15:18:27.693Z",
"stopped": "2021-11-26T15:18:27.693Z",
"brokerage": "Interactive",
"subscription": "string",
"error": "string",
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Read Live Algorithm
Live Management
Read Live Algorithm
The QuantConnect REST API lets you read your live algorithm results from our cloud servers through URL endpoints.
Live Algorithm Statistics
Charts
Portfolio State
Orders
Insights
Logs
API Reference > Live Management > Read Live Algorithm > Live Algorithm Statistics
Read Live Algorithm
Live Algorithm Statistics
Introduction
If a ReadLiveAlgorithmRequest is provided details on a live algorithm are returned. If a ListLiveAlgorithmsRequest is passed get
a list of live running algorithms.
Request
Dynamic arguement to specify whether seeking single project or list response. The /live/read API accepts requests in the
following format:
ReadLiveAlgorithmRequest Model - Request to read out a single algorithm.
projectId
integer
Id of the project to read.
deployId
string
Specific instance Id to read.
Example
{
"projectId": 0,
"deployId": "string"
}
Responses
The /live/read API provides a response in the following format:
200 Success
LiveAlgorithmResults Model - Details a live algorithm from the live/read API endpoint.
message
string
Error message.
status
string
Indicates the status of the algorihtm, i.e. 'Running',
'Stopped'.
deployId
string
Algorithm deployment ID.
cloneId
int
The snapshot project ID for cloning the live development's
source code.
launched
string
Date the live algorithm was launched.
stopped
string
Date the live algorithm was stopped.
brokerage
string
Brokerage used in the live algorithm.
securityTypes
string
Security types present in the live algorithm.
projectName
string
Name of the project the live algorithm is in.
dataCenter
string
Name of the data center where the algorithm is physically
located.
public
bool
Indicates if the algorithm is being live shared.
files
ProjectFile Array
Files present in the project in which the algorithm is.
runtimeStatistics
RuntimeStatistics object
Runtime banner/updating statistics in the title banner of the
live algorithm GUI. It can be empty if the algorithm is not
running.
charts
ChartSummary object
Chart updates for the live algorithm since the last result
packet.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
E
x
a
m
ple
{
"
m
e
s
s
a
g
e
": "
s
t
r
i
n
g
"
,
"
s
t
a
t
u
s
": "
s
t
r
i
n
g
"
,
"
d
e
p
l
o
y
I
d
": "
s
t
r
i
n
g
"
,
"
c
l
o
n
e
I
d
": ,
"
l
a
u
n
c
h
e
d
": "
s
t
r
i
n
g
"
,
"
s
t
o
p
p
e
d
": "
s
t
r
i
n
g
"
,
"
b
r
o
k
e
r
a
g
e
": "
s
t
r
i
n
g
"
,
"
s
e
c
u
r
i
t
y
T
y
p
e
s
": "
s
t
r
i
n
g
"
,
"
p
r
o
j
e
c
t
N
a
m
e
": "
s
t
r
i
n
g
"
,
"
d
a
t
a
C
e
n
t
e
r
": "
s
t
r
i
n
g
"
,
"
p
u
b
l
i
c
": ,
"
f
i
l
e
s
": [
{
"
i
d
": ,
"
p
r
o
j
e
c
t
I
d
": ,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
c
o
n
t
e
n
t
": "
s
t
r
i
n
g
"
,
"
m
o
d
i
f
i
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
o
p
e
n
": t
r
u
e
,
"
i
s
L
i
b
r
a
r
y
": t
r
u
e
}
]
,
"
r
u
n
t
i
m
e
S
t
a
t
i
s
t
i
c
s
": {
"
E
q
u
i
t
y
": "
$
1
0
0.0
0
"
,
"
F
e
e
s
": "
-
$
1
0
0.0
0
"
,
"
H
o
l
d
i
n
g
s
": "
$
1
0
0.0
0
"
,
"
N
e
t
P
r
o
f
i
t
": "
$
1
0
0.0
0
"
,
"
P
r
o
b
a
b
i
l
i
s
t
i
c
S
h
a
r
p
e
R
a
t
i
o
": "
5
0.0
0
%
"
,
"
R
e
t
u
r
n
": "
5
0.0
0
%
"
,
"
U
n
r
e
a
l
i
z
e
d
": "
$
1
0
0.0
0
"
,
"
V
o
l
u
m
e
": "
$
1
0
0.0
0
"
}
,
"
c
h
a
r
t
s
": {
"
n
a
m
e
": "
s
t
r
i
n
g
"
}
,
"
s
u
c
c
e
s
s
": t
r
u
e
,
"
e
r
r
o
r
s
": [
"
s
t
r
i
n
g
"
]
}
ProjectFile Model - File for a project.
id
int
ID of the project file. This can also be null.
projectId
int
ID of the project.
name
string
Name of a project file.
content
string
Contents of the project file.
modified
string($date-time)
DateTime project file was modified.
open
boolean
Indicates if the project file is open or not.
isLibrary
boolean
Indicates if the project file is a library or not. It's always
false in live/read and backtest/read.
Example
{
"id": ,
"projectId": ,
"name": "string",
"content": "string",
"modified": "2021-11-26T15:18:27.693Z",
"open": true,
"isLibrary": true
}
RuntimeStatistics Model
Equity
string
example:
$100.00
Total portfolio value.
Fees
string
example: -$100.00
Transaction fee.
Holdings
string
example:
$100.00
Equity value of security holdings.
Net Profit
string
example:
$100.00
Net profit.
Probabilistic Sharpe Ratio
string
example: 50.00%
Probabilistic Sharpe Ratio.
Return
string
example: 50.00%
Return.
Unrealized
string
example:
$100.00
Unrealized profit/loss.
Volume
string
example:
$100.00
Total transaction volume.
Example
{
"Equity": "$100.00",
"Fees": "-$100.00",
"Holdings": "$100.00",
"Net Profit": "$100.00",
"Probabilistic Sharpe Ratio": "50.00%",
"Return": "50.00%",
"Unrealized": "$100.00",
"Volume": "$100.00"
}
ChartSummary Model - Contains the names of all charts
name
string
Name of the Chart.
Example
{
"name": "string"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Read Live Algorithm > Charts
Read Live Algorithm
Charts
Introduction
Read a chart from a live algorithm.
Request
Request body to obtain a chart from a live algorithm. The /live/chart/read API accepts requests in the following format:
ReadLiveChartRequest Model - Request to body to obtain a chart from a live algorithm.
projectId
integer
example: 12345678
Project ID of the request.
name
string
example: Strategy Equity
The requested chart name.
count
integer
example: 100
The number of data points to request.
start
integer
example: 1717801200
The Utc start seconds timestamp of the request.
end
integer
example: 1743462000
The Utc end seconds timestamp of the request.
Example
{
"projectId": 12345678,
"name": "Strategy Equity",
"count": 100,
"start": 1717801200,
"end": 1743462000
}
Responses
The /live/chart/read API provides a response in the following format:
200 Success
LoadingChartResponse Model - Response when the requested chart is being generated.
progress
number
Loading percentage of the chart generation process.
status
string
example:
loading
Status of the chart generation process.
success
boolean
Indicate if the API request was successful.
Example
{
"progress": 0,
"status": "loading",
"success": true
}
ReadChartResponse Model - Response with the requested chart from a live algorithm or backtest.
chart
Chart object
Single Parent Chart Object for Custom Charting.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"chart": {
"name": "string",
"chartType": "Overlay",
"series": {
"name": "string",
"unit": "string",
"index": 0,
"values": [
"object"
],
"seriesType": "Line",
"color": "string",
"scatterMarkerSymbol": "none"
}
},
"success": true,
"errors": [
"string"
]
}
Chart Model - Single Parent Chart Object for Custom Charting.
name
string
Name of the Chart.
chartType
string Enum
Type of the Chart, Overlayed or Stacked. Options :
['Overlay', 'Stacked']
series
Series object
List of Series Objects for this Chart.
Example
{
"name": "string",
"chartType": "Overlay",
"series": {
"name": "string",
"unit": "string",
"index": 0,
"values": [
"object"
],
"seriesType": "Line",
"color": "string",
"scatterMarkerSymbol": "none"
}
}
Series Model - Chart Series Object - Series data and properties for a chart.
name
string
Name of the series.
unit
string
Axis for the chart series.
index
integer
Index/position of the series on the chart.
values
object Array
Values for the series plot. These values are assumed to be
in ascending time order (first points earliest, last points
latest).
seriesType
string Enum
Chart type for the series. Options : ['Line', 'Scatter',
'Candle', 'Bar', 'Flag', 'StackedArea', 'Pie', 'Treemap']
color
string
Color the series.
scatterMarkerSymbol
string Enum
Shape or symbol for the marker in a scatter plot. Options :
['none', 'circle', 'square', 'diamond', 'triangle', 'triangledown']
Example
{
"name": "string",
"unit": "string",
"index": 0,
"values": [
"object"
],
"seriesType": "Line",
"color": "string",
"scatterMarkerSymbol": "none"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Read Live Algorithm > Portfolio State
Read Live Algorithm
Portfolio State
Introduction
Read out the portfolio state of a live algorithm.
Request
Fetch the live portfolio state for the project Id provided. The /live/portfolio/read API accepts requests in the following
format:
ReadLivePortfolioRequest Model - Request to read the portfolio state from a live algorithm.
projectId
integer
Id of the project from which to read the live algorithm.
Example
{
"projectId": 0
}
Responses
The /live/portfolio/read API provides a response in the following format:
200 Success
LivePortfolioResponse Model - Contains holdings and cash of the live algorithm in the request criteria.
portfolio
Portfolio object
Portfolio object with the holdings and cash information.
Example
{
"portfolio": {
"holdings": {
"AAPL R735QTJ8XC9X": {
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"type": "Base",
"currencySymbol": "$",
"averagePrice": 0,
"quantity": 0,
"marketPrice": 0,
"conversionRate": 0,
"marketValue": 0,
"unrealizedPnl": 0
}
},
"cash": {
"USD": {
"symbol": "string",
"amount": 0,
"conversionRate": 0,
"currencySymbol": ,
"valueInAccountCurrency": 0
}
}
}
}
Portfolio Model - Portfolio object with the holdings and cash information.
holdings
object
Dictionary of algorithm holdings information.
cash
object
Dictionary of algorithm cash currencies information.
Example
{
"holdings": {
"AAPL R735QTJ8XC9X": {
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"type": "Base",
"currencySymbol": "$",
"averagePrice": 0,
"quantity": 0,
"marketPrice": 0,
"conversionRate": 0,
"marketValue": 0,
"unrealizedPnl": 0
}
},
"cash": {
"USD": {
"symbol": "string",
"amount": 0,
"conversionRate": 0,
"currencySymbol": ,
"valueInAccountCurrency": 0
}
}
}
Holding Model - Live results object class for packaging live result data.
symbol
Symbol object
Represents a unique security identifier. This is made of two
components, the unique SID and the Value. The value is the
current ticker symbol while the SID is constant over the life
of a security.
type
string Enum
Type of tradable security / underlying asset. Options :
['Base', 'Equity', 'Option', 'Commodity', 'Forex', 'Future',
'Cfd', 'Crypto']
currencySymbol
string
example: $
The currency symbol of the holding.
averagePrice
number
Average Price of our Holding in the currency the symbol is
traded in.
quantity
number
Quantity of the Symbol we hold.
marketPrice
number
Current Market Price of the Asset in the currency the
symbol is traded in.
conversionRate
number
Current market conversion rate into the account currency.
marketValue
number
Current market value of the holding.
unrealizedPnl
number
Current unrealized P/L of the holding.
Example
{
"symbol": {
"value": "string",
"id": "string",
"permtick": "string"
},
"type": "Base",
"currencySymbol": "$",
"averagePrice": 0,
"quantity": 0,
"marketPrice": 0,
"conversionRate": 0,
"marketValue": 0,
"unrealizedPnl": 0
}
Symbol Model - Represents a unique security identifier. This is made of two components, the unique SID and the Value. The
value is the current ticker symbol while the SID is constant over the life of a security.
value
string
The current symbol for this ticker.
id
string
The security identifier for this symbol.
permtick
string
The current symbol for this ticker.
Example
{
"value": "string",
"id": "string",
"permtick": "string"
}
Cash Model - Represents a holding of a currency in cash.
symbol
string
Gets the symbol used to represent this cash.
amount
number
Gets or sets the amount of cash held.
conversionRate
number
The currency conversion rate to the account base currency.
currencySymbol
object
The symbol of the currency, such as $.
valueInAccountCurrency
number
The value of the currency cash in the account base
currency.
Example
{
"symbol": "string",
"amount": 0,
"conversionRate": 0,
"currencySymbol": ,
"valueInAccountCurrency": 0
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Read Live Algorithm > Orders
Read Live Algorithm
Orders
Introduction
Read out the orders of a live algorithm.
Request
Fetch the orders of a live algorithm for the project Id and steps provided. The /live/orders/read API accepts requests in the
following format:
ReadLiveOrdersRequest Model - Request to read orders from a live algorithm.
start
integer
Starting index of the orders to be fetched. Required if end >
100.
end
integer
Last index of the orders to be fetched. Note that end - start
must be less than 100.
projectId
integer
Id of the project from which to read the live algorithm.
Example
{
"start": 0,
"end": 0,
"projectId": 0
}
Responses
The /live/orders/read API provides a response in the following format:
200 Success
LiveOrdersResponse Model - Contains orders and the number of orders of the live algorithm in the request criteria.
orders
Order Array
Collection of orders.
length
integer
Total number of returned orders.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
E
x
a
m
ple
{
"
o
r
d
e
r
s
": [
{
"
i
d
": 0
,
"
c
o
n
t
i
n
g
e
n
t
I
d
": 0
,
"
b
r
o
k
e
r
I
d
": [
"
s
t
r
i
n
g
"
]
,
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
l
i
m
i
t
P
r
i
c
e
": ,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
s
t
o
p
T
r
i
g
g
e
r
e
d
": ,
"
p
r
i
c
e
": 0
,
"
p
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
t
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
F
i
l
l
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
U
p
d
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
a
n
c
e
l
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
y
p
e
": 0
=
M
a
r
k
e
t
,
"
s
t
a
t
u
s
": 0
=
N
e
w
,
"
t
a
g
": "
s
t
r
i
n
g
"
,
"
s
e
c
u
r
i
t
y
T
y
p
e
": 0
=
B
a
s
e
,
"
d
i
r
e
c
t
i
o
n
": 0
=
B
u
y
,
"
v
a
l
u
e
": 0
,
"
o
r
d
e
r
S
u
b
m
i
s
s
i
o
n
D
a
t
a
": {
"
b
i
d
P
r
i
c
e
": 0
,
"
a
s
k
P
r
i
c
e
": 0
,
"
l
a
s
t
P
r
i
c
e
": 0
}
,
"
i
s
M
a
r
k
e
t
a
b
l
e
": t
r
u
e
,
"
p
r
o
p
e
r
t
i
e
s
": {
"
t
i
m
e
I
n
F
o
r
c
e
": 0
=
G
o
o
d
T
i
l
C
a
n
c
e
l
e
d
}
,
"
e
v
e
n
t
s
": [
{
"
a
l
g
o
r
i
t
h
m
I
d
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
V
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
P
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
,
"
o
r
d
e
r
I
d
": 0
,
"
o
r
d
e
r
E
v
e
n
t
I
d
": 0
,
"
i
d
": 0
,
"
s
t
a
t
u
s
": "
n
e
w
"
,
"
o
r
d
e
r
F
e
e
A
m
o
u
n
t
": 0
,
"
o
r
d
e
r
F
e
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
P
r
i
c
e
": 0
,
"
f
i
l
l
P
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
Q
u
a
n
t
i
t
y
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
e
s
s
a
g
e
": "
s
t
r
i
n
g
"
,
"
i
s
A
s
s
i
g
n
m
e
n
t
": t
r
u
e
,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
i
m
e
": 0
,
"
i
s
I
n
T
h
e
M
o
n
e
y
": }
]
,
"
t
r
a
i
l
i
n
g
A
m
o
u
n
t
": 0
,
"
t
r
a
i
l
i
n
g
P
e
r
c
e
n
t
a
g
e
": ,
"
g
r
o
u
p
O
r
d
e
r
M
a
n
a
g
e
r
": {
"
i
d
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
c
o
u
n
t
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
o
r
d
e
r
I
d
s
": [
"
i
n
t
e
g
e
r
"
]
,
"
d
i
r
e
c
t
i
o
n
": 0
}
,
"
t
r
i
g
g
e
r
P
r
i
c
e
": 0
,
"
t
r
i
g
g
e
r
T
o
u
c
h
e
d
": }
}
],
"length": 0,
"success": true,
"errors": [
"string"
]
}
Order Model - Order struct for placing new trade.
id
integer
Order ID.
contingentId
integer
Order Id to process before processing this order.
brokerId
string Array
Brokerage Id for this order for when the brokerage splits
orders into multiple pieces.
symbol
Symbol object
Represents a unique security identifier. This is made of two
components, the unique SID and the Value. The value is the
current ticker symbol while the SID is constant over the life
of a security.
limitPrice
nummber
Limit price of the Order.
stopPrice
number
Stop price of the Order.
stopTriggered
bool
Indicates if the stop price has been reached, so the limit
order has been triggered.
price
number
Price of the Order.
priceCurrency
string
Currency for the order price.
time
string($date-time)
Gets the utc time the order was created.
createdTime
string($date-time)
Gets the utc time this order was created. Alias for Time.
lastFillTime
string($date-time)
Gets the utc time the last fill was received, or null if no fills
have been received.
lastUpdateTime
string($date-time)
Gets the utc time this order was last updated, or null if the
order has not been updated.
canceledTime
string($date-time)
Gets the utc time this order was canceled, or null if the
order was not canceled.
quantity
number
Number of shares to execute.
type
integer Enum
Order type. Options : ['0 = Market', '1 = Limit', '2 =
StopMarket', '3 = StopLimit', '4 = MarketOnOpen', '5 = MarketOnClose', '6 = OptionExercise', '7 = LimitIfTouched',
'8 = ComboMarket', '9 = ComboLimit', '10 =
ComboLegLimit', '11 = TrailingStop']
status
integer Enum
Status of the Order. Options : ['0 = New', '1 = Submitted', '2
= PartiallyFilled', '3 = Filled', '5 = Canceled', '6 = None', '7 =
Invalid', '8 = CancelPending', '9 = UpdateSubmitted']
tag
string
Tag the order with some custom data.
securityType
integer Enum
Type of tradable security / underlying asset. Options : ['0 =
Base', '1 = Equity', '2 = Option', '3 = Commodity', '4 =
Forex', '5 = Future', '6 = Cfd', '7 = Crypto']
direction
integer Enum
Order Direction Property based off Quantity. Options : ['0 =
Buy', '1 = Sell', '2 = Hold']
value
number
Gets the executed value of this order. If the order has not
yet filled, then this will return zero.
orderSubmissionData
OrderSubmissionData object
Stores time and price information available at the time an
order was submitted.
isMarketable
boolean
Returns true if the order is a marketable order.
properties
OrderProperties object
Additional properties of the order.
events
OrderEvent Array
The order events.
trailingAmount
number
Trailing amount for a trailing stop order.
trailingPercentage
bool
Determines whether the trailingAmount is a percentage or
an absolute currency value.
groupOrderManager
GroupOrderManager object
Manager of a group of orders.
triggerPrice
number
The price which, when touched, will trigger the setting of a
limit order at limitPrice.
triggerTouched
bool
Whether or not the triggerPrice has been touched.
E
x
a
m
ple
{
"
i
d
": 0
,
"
c
o
n
t
i
n
g
e
n
t
I
d
": 0
,
"
b
r
o
k
e
r
I
d
": [
"
s
t
r
i
n
g
"
]
,
"
s
y
m
b
o
l
": {
"
v
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
i
d
": "
s
t
r
i
n
g
"
,
"
p
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
}
,
"
l
i
m
i
t
P
r
i
c
e
": ,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
s
t
o
p
T
r
i
g
g
e
r
e
d
": ,
"
p
r
i
c
e
": 0
,
"
p
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
t
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
r
e
a
t
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
F
i
l
l
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
l
a
s
t
U
p
d
a
t
e
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
c
a
n
c
e
l
e
d
T
i
m
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
y
p
e
": 0
=
M
a
r
k
e
t
,
"
s
t
a
t
u
s
": 0
=
N
e
w
,
"
t
a
g
": "
s
t
r
i
n
g
"
,
"
s
e
c
u
r
i
t
y
T
y
p
e
": 0
=
B
a
s
e
,
"
d
i
r
e
c
t
i
o
n
": 0
=
B
u
y
,
"
v
a
l
u
e
": 0
,
"
o
r
d
e
r
S
u
b
m
i
s
s
i
o
n
D
a
t
a
": {
"
b
i
d
P
r
i
c
e
": 0
,
"
a
s
k
P
r
i
c
e
": 0
,
"
l
a
s
t
P
r
i
c
e
": 0
}
,
"
i
s
M
a
r
k
e
t
a
b
l
e
": t
r
u
e
,
"
p
r
o
p
e
r
t
i
e
s
": {
"
t
i
m
e
I
n
F
o
r
c
e
": 0
=
G
o
o
d
T
i
l
C
a
n
c
e
l
e
d
}
,
"
e
v
e
n
t
s
": [
{
"
a
l
g
o
r
i
t
h
m
I
d
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
V
a
l
u
e
": "
s
t
r
i
n
g
"
,
"
s
y
m
b
o
l
P
e
r
m
t
i
c
k
": "
s
t
r
i
n
g
"
,
"
o
r
d
e
r
I
d
": 0
,
"
o
r
d
e
r
E
v
e
n
t
I
d
": 0
,
"
i
d
": 0
,
"
s
t
a
t
u
s
": "
n
e
w
"
,
"
o
r
d
e
r
F
e
e
A
m
o
u
n
t
": 0
,
"
o
r
d
e
r
F
e
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
P
r
i
c
e
": 0
,
"
f
i
l
l
P
r
i
c
e
C
u
r
r
e
n
c
y
": "
s
t
r
i
n
g
"
,
"
f
i
l
l
Q
u
a
n
t
i
t
y
": 0
,
"
d
i
r
e
c
t
i
o
n
": "
s
t
r
i
n
g
"
,
"
m
e
s
s
a
g
e
": "
s
t
r
i
n
g
"
,
"
i
s
A
s
s
i
g
n
m
e
n
t
": t
r
u
e
,
"
s
t
o
p
P
r
i
c
e
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
t
i
m
e
": 0
,
"
i
s
I
n
T
h
e
M
o
n
e
y
": }
]
,
"
t
r
a
i
l
i
n
g
A
m
o
u
n
t
": 0
,
"
t
r
a
i
l
i
n
g
P
e
r
c
e
n
t
a
g
e
": ,
"
g
r
o
u
p
O
r
d
e
r
M
a
n
a
g
e
r
": {
"
i
d
": 0
,
"
q
u
a
n
t
i
t
y
": 0
,
"
c
o
u
n
t
": 0
,
"
l
i
m
i
t
P
r
i
c
e
": 0
,
"
o
r
d
e
r
I
d
s
": [
"
i
n
t
e
g
e
r
"
]
,
"
d
i
r
e
c
t
i
o
n
": 0
}
,
"
t
r
i
g
g
e
r
P
r
i
c
e
": 0
,
"
t
r
i
g
g
e
r
T
o
u
c
h
e
d
": }
Symbol Model - Represents a unique security identifier. This is made of two components, the unique SID and the Value. The
value is the current ticker symbol while the SID is constant over the life of a security.
value
string
The current symbol for this ticker.
id
string
The security identifier for this symbol.
permtick
string
The current symbol for this ticker.
Example
{
"value": "string",
"id": "string",
"permtick": "string"
}
OrderSubmissionData Model - Stores time and price information available at the time an order was submitted.
bidPrice
number
The bid price at an order submission time.
askPrice
number
The ask price at an order submission time.
lastPrice
number
The current price at an order submission time.
Example
{
"bidPrice": 0,
"askPrice": 0,
"lastPrice": 0
}
OrderProperties Model - Additional properties of the order.
timeInForce
object Enum
Defines the length of time over which an order will continue
working before it is cancelled. Options : ['0 =
GoodTilCanceled', '1 = Day', '2 = GoodTilDate']
Example
{
"timeInForce": 0 = GoodTilCanceled
}
OrderEvent Model - Change in an order state applied to user algorithm portfolio
algorithmId
string
Algorithm Id, BacktestId or DeployId.
symbol
string
Easy access to the order symbol associated with this event.
symbolValue
string
The current symbol for this ticker; It is a user friendly
symbol representation.
symbolPermtick
string
The original symbol used to generate this symbol.
orderId
integer
Id of the order this event comes from.
orderEventId
integer
The unique order event id for each order.
id
integer
The unique order event Id for each order.
status
string Enum
Status of the Order. Options : ['new', 'submitted',
'partiallyFilled', 'filled', 'canceled', 'none', 'invalid',
'cancelPending', 'updateSubmitted']
orderFeeAmount
number
The fee amount associated with the order.
orderFeeCurrency
string
The fee currency associated with the order.
fillPrice
number
Fill price information about the order.
fillPriceCurrency
string
Currency for the fill price.
fillQuantity
number
Number of shares of the order that was filled in this event.
direction
string
Order direction.
message
string
Any message from the exchange.
isAssignment
boolean
True if the order event is an assignment.
stopPrice
number
The current stop price.
limitPrice
number
The current limit price.
quantity
number
The current order quantity.
tim
e
i
n
t
e
g
e
r
T
h
e
tim
e
o
f
t
his
e
v
e
n
t in
u
nix
tim
e
s
t
a
m
p. isInTheMoney bool True if the order event's option is In-The
-
M
o
n
e
y
(IT
M
). Example { "algorithmId": "string", "symbol": "string", "symbolValue": "string", "symbolPermtick": "string", "orderId": 0, "orderEventId": 0, "id": 0, "status": "new", "orderFeeAmount": 0, "orderFeeCurrency": "string", "fillPrice": 0, "fillPriceCurrency": "string", "fillQuantity": 0, "direction": "string", "message": "string", "isAssignment": true, "stopPrice": 0, "limitPrice": 0, "quantity": 0, "time": 0, "isInTheMoney": }
GroupOrderManager Model - Manager of a group of orders.
id
integer
The unique order group Id.
quantity
number
The group order quantity.
count
integer
The total order count associated with this order group.
limitPrice
number
The limit price associated with this order group if any.
orderIds
integer Array
The order Ids in this group.
direction
integer
Order Direction Property based off Quantity.
Example
{
"id": 0,
"quantity": 0,
"count": 0,
"limitPrice": 0,
"orderIds": [
"integer"
],
"direction": 0
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Read Live Algorithm > Insights
Read Live Algorithm
Insights
Introduction
Read out the insights of a live algorithm.
Request
Fetch the insights of a live algorithm for the project Id and steps provided. The /live/insights/read API accepts requests in
the following format:
ReadLiveInsightsRequest Model - Request to read insights from a live algorithm.
start
integer
Starting index of the insights to be fetched. Required if end
> 100.
end
integer
Last index of the insights to be fetched. Note that end -
start must be less than 100.
projectId
integer
Id of the project from which to read the live algorithm.
Example
{
"start": 0,
"end": 0,
"projectId": 0
}
Responses
The /live/insights/read API provides a response in the following format:
200 Success
LiveInsightsResponse Model - Contains insights and the number of insights of the live algorithm in the request criteria.
insights
Insight Array
Collection of insights.
length
integer
Total number of returned insights.
success
boolean
Indicate if the API request was successful.
Example
{
"insights": [
{
"id": "string",
"groupId": "string",
"sourceModel": "string",
"generatedTime": "string",
"createdTime": 0,
"closeTime": 0,
"symbol": "string",
"ticker": "string",
"type": "price",
"reference": "string",
"referenceValueFinal": "string",
"direction": "down",
"period": 0,
"magnitude": 0,
"confidence": 0,
"weight": 0,
"scoreIsFinal": ,
"scoreDirection": 0,
"scoreMagnitude": 0,
"estimatedValue": 0,
"tag": "2021-11-26T15:18:27.693Z"
}
],
"length": 0,
"success": true
}
Insight Model - Insight struct for emitting new prediction.
id
string
Insight ID.
groupId
string
ID of the group of insights.
sourceModel
string
Name of the model that sourced the insight.
generatedTime
string
Gets the utc unixtime this insight was generated.
createdTime
number
Gets the utc unixtime this insight was created.
closeTime
number
Gets the utc unixtime this insight was closed.
symbol
string
Gets the symbol ID this insight is for.
ticker
string
Gets the symbol ticker this insight is for.
type
string Enum
Gets the type of insight, for example, price or volatility.
Options : ['price', 'volatility']
reference
string
Gets the initial reference value this insight is predicting
against.
referenceValueFinal
string
Gets the final reference value, used for scoring, this insight
is predicting against.
direction
string Enum
Gets the predicted direction, down, flat or up. Options :
['down', 'flat', 'up']
period
number
Gets the period, in seconds, over which this insight is
expected to come to fruition.
magnitude
number
Gets the predicted percent change in the insight type
(price/volatility). This value can be null.
confidence
number
Gets the confidence in this insight. This value can be null.
weight
number
Gets the portfolio weight of this insight. This value can be
null.
scoreIsFinal
bool
Gets whether or not this is the insight's final score.
scoreDirection
number
Gets the direction score.
scoreMagnitude
number
Gets the magnitude score.
estimatedValue
number
Gets the estimated value of this insight in the account
currency.
tag
string($float)
The insight's tag containing additional information.
Example
{
"id": "string",
"groupId": "string",
"sourceModel": "string",
"generatedTime": "string",
"createdTime": 0,
"closeTime": 0,
"symbol": "string",
"ticker": "string",
"type": "price",
"reference": "string",
"referenceValueFinal": "string",
"direction": "down",
"period": 0,
"magnitude": 0,
"confidence": 0,
"weight": 0,
"scoreIsFinal": ,
"scoreDirection": 0,
"scoreMagnitude": 0,
"estimatedValue": 0,
"tag": "2021-11-26T15:18:27.693Z"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Read Live Algorithm > Logs
Read Live Algorithm
Logs
Introduction
Get the logs of a specific live algorithm.
Request
Information about the algorithm to read live logs from. The /live/logs/read API accepts requests in the following format:
ReadLiveLogsRequest Model - Request to read the logs of a specific algorithm. The difference between 'startLine' and
'endLine' must be smaller than 250, else an error will be thrown.
format
object
example: json
Format of the log results.
projectId
integer
Project Id of the live running algorithm.
algorithmId
string
Deploy Id (Algorithm Id) of the live running algorithm.
startLine
integer
Start line of logs to read.
endLine
integer
End line of logs to read.
Example
{
"format": "json",
"projectId": 0,
"algorithmId": "string",
"startLine": 0,
"endLine": 0
}
Responses
The /live/logs/read API provides a response in the following format:
200 Success
ReadLiveLogsResponse Model - Logs from a live algorithm.
logs
string Array
List of logs from the live algorithm.
length
integer
Total amount of rows in the logs.
deploymentOffset
integer
Amount of log rows before the current deployment.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"logs": [
"string"
],
"length": 0,
"deploymentOffset": 0,
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Update Live Algorithm
Live Management
Update Live Algorithm
The QuantConnect REST API lets you update your live algorithms on our cloud servers through URL endpoints.
Liquidate Live Portfolio
Stop Live Algorithm
API Reference > Live Management > Update Live Algorithm > Liquidate Live Portfolio
Update Live Algorithm
Liquidate Live Portfolio
Introduction
Liquidate a live algorithm from the specified project and deployId.
Request
Information about the live algorithm to liquidate. The /live/update/liquidate API accepts requests in the following format:
LiquidateLiveAlgorithmRequest Model - Request to liquidate a live algorithm.
projectId
integer
Project Id for the live instance we want to liquidate.
Example
{
"projectId": 0
}
Responses
The /live/update/liquidate API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Update Live Algorithm > Stop Live Algorithm
Update Live Algorithm
Stop Live Algorithm
Introduction
Stop a live algorithm from the specified project and deployId.
Request
Information about the project to delete. The /live/update/stop API accepts requests in the following format:
StopLiveAlgorithmRequest Model - Request to stop a live algorithm.
projectId
integer
Project Id for the live instance we want to stop.
Example
{
"projectId": 0
}
Responses
The /live/update/stop API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > List Live Algorithms
Live Management
List Live Algorithms
Introduction
Returns a list of live running algorithms.
Request
Request body to obtain a list of live running algorithms. The /live/list API accepts requests in the following format:
ListLiveAlgorithmsRequest Model - Request for a list of live running algorithms.
status
string Enum
States of a live deployment. Options : ['DeployError',
'InQueue', 'Running', 'Stopped', 'Liquidated', 'Deleted',
'Completed', 'RuntimeError', 'Invalid', 'LoggingIn',
'Initializing', 'History']
start
number
Earliest launched time of the algorithms in UNIX timestamp.
end
number
Latest launched time of the algorithms in UNIX timestamp.
Example
{
"status": "DeployError",
"start": 0,
"end": 0
}
Responses
The /live/list API provides a response in the following format:
200 Success
LiveAlgorithmListResponse Model - List of the live algorithms running which match the requested status.
live
LiveAlgorithm Array
Algorithm list matching the requested status.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"live": [
{
"projectId": 0,
"deployId": "string",
"status": "DeployError",
"launched": "2021-11-26T15:18:27.693Z",
"stopped": "2021-11-26T15:18:27.693Z",
"brokerage": "Interactive",
"subscription": "string",
"error": "string",
"success": true,
"errors": [
"string"
]
}
],
"success": true,
"errors": [
"string"
]
}
LiveAlgorithm Model - Live algorithm instance result from the QuantConnect Rest API.
projectId
integer
Project Id for the live instance.
deployId
string
Unique live algorithm deployment identifier (similar to a
backtest id).
status
string Enum
States of a live deployment. Options : ['DeployError',
'InQueue', 'Running', 'Stopped', 'Liquidated', 'Deleted',
'Completed', 'RuntimeError', 'Invalid', 'LoggingIn',
'Initializing', 'History']
launched
string($date-time)
Datetime the algorithm was launched in UTC.
stopped
string($date-time)
Datetime the algorithm was stopped in UTC, null if its still
running.
brokerage
string Enum
Brokerage. Options : ['Interactive', 'FXCM', 'Oanda',
'Tradier', 'PaperBrokerage', 'Alpaca', 'Bitfinex', 'Binance',
'Coinbase']
subscription
string
Chart we're subscribed to.
error
string
Live algorithm error message from a crash or algorithm
runtime error.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"projectId": 0,
"deployId": "string",
"status": "DeployError",
"launched": "2021-11-26T15:18:27.693Z",
"stopped": "2021-11-26T15:18:27.693Z",
"brokerage": "Interactive",
"subscription": "string",
"error": "string",
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Live Commands
Live Management
Live Commands
The QuantConnect REST API send commands to live algorithms on our cloud servers through URL endpoints.
Create Live Command
Broadcast Live Command
API Reference > Live Management > Live Commands > Create Live Command
Live Commands
Create Live Command
Introduction
Create a live command.
Request
Sends a command to a live deployment to trigger an action such as placing orders. The /live/commands/create API accepts
requests in the following format:
CreateLiveCommandRequest Model - Request to create a live command.
projectId
integer
example: 19626262
Project for the live instance we want to run the command
against.
command
object
example: {'$type': 'OrderCommand', 'symbol': {'id': 'BTCUSD
2XR', 'value': 'BTCUSD'}, 'order_type': 'market', 'quantity':
'0.1', 'limit_price': 0, 'stop_price': 0, 'tag': ''}
The command to run.
Example
{
"projectId": 19626262,
"command": {
"$type": "OrderCommand",
"symbol": {
"id": "BTCUSD 2XR",
"value": "BTCUSD"
},
"order_type": "market",
"quantity": "0.1",
"limit_price": 0,
"stop_price": 0,
"tag": ""
}
}
Responses
The /live/commands/create API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Live Management > Live Commands > Broadcast Live Command
Live Commands
Broadcast Live Command
Introduction
Broadcast a live command to all live algorithms in the organization.
Request
Broadcasts a command to all live deployments in the organization. The /live/commands/broadcast API accepts requests in the
following format:
BroadcastLiveCommandRequest Model - Request to create a live command.
organizationId
string
example: 5cad178b20a1d52567b534553413b691
Organization Id of the projects we would like to broadcast
the command to.
excludeProjectId
integer
example: 19626262
Project for the live instance we want to exclude from the
broadcast list. If null, all projects will be included.
command
object
example: {'$type': 'OrderCommand', 'symbol': {'id': 'BTCUSD
2XR', 'value': 'BTCUSD'}, 'order_type': 'market', 'quantity':
'0.1', 'limit_price': 0, 'stop_price': 0, 'tag': ''}
The command to run.
Example
{
"organizationId":
"5cad178b20a1d52567b534553413b691",
"excludeProjectId": 19626262,
"command": {
"$type": "OrderCommand",
"symbol": {
"id": "BTCUSD 2XR",
"value": "BTCUSD"
},
"order_type": "market",
"quantity": "0.1",
"limit_price": 0,
"stop_price": 0,
"tag": ""
}
}
Responses
The /live/commands/broadcast API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Optimization Management
API Reference
Optimization Management
API Reference > Optimization Management > Create Optimization
Optimization Management
Create Optimization
Introduction
Create an optimization with the specified parameters.
Request
Project, compile and optimization parameters for launching an optimization job. The /optimizations/create API accepts
requests in the following format:
CreateOptimizationRequest Model - Request to create an optimization job.
projectId
integer
Project ID of the project the optimization belongs to.
name
string
Name of the optimization.
target
string
Target of the optimization.
targetTo
string
example: max or
min
Target extremum of the optimization.
targetValue
float
Optimization target value.
strategy
string
example:
QuantConnect.Optimizer.Strategies.GridSearchOptimizationStrategy
Optimization strategy.
compileId
string
Optimization compile ID.
parameters
OptimizationParameter Array
Optimization parameters.
constraints
OptimizationConstraint Array
Optimization constraints.
estimatedCost
float
example:
10
Estimated cost for optimization.
nodeType
string Enum
example: O2-8
Optimization node type. Options : ['O2-8', 'O4-12', 'O8-16']
parallelNodes
integer
example: 4
Number of parallel nodes for optimization.
Example
{
"projectId": 0,
"name": "string",
"target": "string",
"targetTo": "max or min",
"targetValue": ,
"strategy":
"QuantConnect.Optimizer.Strategies.GridSearchOptimizationStrategy"compileId": "string",
"parameters": [
{
"name": "rsi_period",
"min": 10,
"max": 20,
"step": 1,
"minStep": 1
}
],
"constraints": [
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"operator": "greater",
"targetValue": 1
}
],
"estimatedCost": 10,
"nodeType": "O2-8",
"parallelNodes": 4
}
OptimizationParameter Model
name
string
example: rsi_period
Name of optimization parameter.
min
float
example:
10
Minimum value of optimization parameter, applicable for
boundary conditions.
max
float
example:
20
Maximum value of optimization parameter, applicable for
boundary conditions.
step
float
example: 1
Movement, should be positive.
minStep
float
example: 1
Minimal possible movement for current parameter, should
be positive. Used by
Strategies.EulerSearchOptimizationStrategy to
determine when this parameter can no longer be optimized.
Example
{
"name": "rsi_period",
"min": 10,
"max": 20,
"step": 1,
"minStep": 1
}
OptimizationConstraint Model
target
string
example: TotalPerformance.PortfolioStatistics.SharpeRatio
Property we want to track.
operator
string
example:
greater
The target comparison operation.
targetValue
float
example: 1
The value of the property we want to track.
Example
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"operator": "greater",
"targetValue": 1
}
Responses
The /optimizations/create API provides a response in the following format:
200 Success
ListOptimizationResponse Model - Response received when listing optimizations of a project.
optimizations
CreateOptimizationResponse Array
Collection of summarized optimization objects.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"optimizations": [
{
"optimizationId": "string",
"projectId": "string",
"name": "string",
"status": "New",
"nodeType": "O2-8",
"criterion": {
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"extremum": "max or min",
"targetValue": 1
},
"created": "2021-11-26T15:18:27.693Z",
"psr": 0,
"sharpeRatio": 0,
"trades": 0,
"cloneId": 0,
"outOfSampleDays": 0,
"outOfSampleMaxEndDate": "2021-11-
26T15:18:27.693Z",
"parameters": [
"object"
]
}
],
"success": true,
"errors": [
"string"
]
}
CreateOptimizationResponse Model - Response received when launching an optimization job.
optimizationId
string
Optimization ID.
projectId
string
Project ID of the project the optimization belongs to.
name
string
Name of the optimization.
status
string Enum
Status of the optimization. Options : ['New', 'Aborted',
'Running', 'Completed']
nodeType
string Enum
example: O2-8
Optimization node type. Options : ['O2-8', 'O4-12', 'O8-16']
criterion
OptimizationTarget object
/.
created
string($date-time)
Date when this optimization was created.
psr
number
Price-sales ratio stastic.
sharpeRatio
number
Sharpe ratio statistic.
trades
integer
Number of trades.
cloneId
integer
ID of project, were this current project was originally
cloned.
outOfSampleDays
integer
Number of days of out of sample days.
outOfSampleMaxEndDate
string($date-time)
End date of out of sample data.
parameters
object Array
Parameters used in this optimization.
E
x
a
m
ple
{
"
o
p
t
i
m
i
z
a
t
i
o
n
I
d
": "
s
t
r
i
n
g
"
,
"
p
r
o
j
e
c
t
I
d
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
s
t
a
t
u
s
": "
N
e
w
"
,
"
n
o
d
e
T
y
p
e
": "
O
2
-
8
"
,
"
c
r
i
t
e
r
i
o
n
": {
"
t
a
r
g
e
t
": "TotalPerform
a
n
c
e.P
o
r
t
f
o
l
i
o
S
t
a
t
i
s
t
i
c
s.S
h
a
r
p
e
R
a
t
i
o
"
,
"
e
x
t
r
e
m
u
m
": "
m
a
x
o
r
m
i
n
"
,
"
t
a
r
g
e
t
V
a
l
u
e
": 1
}
,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
p
s
r
": 0
,
"
s
h
a
r
p
e
R
a
t
i
o
": 0
,
"
t
r
a
d
e
s
": 0
,
"
c
l
o
n
e
I
d
": 0
,
"
o
u
t
O
f
S
a
m
p
l
e
D
a
y
s
": 0
,
"
o
u
t
O
f
S
a
m
p
l
e
M
a
x
E
n
d
D
a
t
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
p
a
r
a
m
e
t
e
r
s
": [
"
o
b
j
e
c
t
"
]
}
OptimizationTarget Model
target
string
example: TotalPerformance.PortfolioStatistics.SharpeRatio
Property we want to track.
extremum
string
example: max or
min
Defines the direction of optimization.
targetValue
float
example: 1
The value of the property we want to track.
Example
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"extremum": "max or min",
"targetValue": 1
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Optimization Management > Update Optimization
Optimization Management
Update Optimization
Introduction
Updates the name of an optimization.
Request
The /optimizations/update API accepts requests in the following format:
UpdateOptimizationRequest Model - Updates the name of an optimization.
optimizationId
string
Optimization ID we want to update.
name
string
Name we'd like to assign to the optimization.
Example
{
"optimizationId": "string",
"name": "string"
}
Responses
The /optimizations/update API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Optimization Management > Read Optimization
Optimization Management
Read Optimization
Introduction
Read an optimization.
Request
Optimization ID for the optimization we want to read. The /optimizations/read API accepts requests in the following format:
ReadOptimizationRequest Model - Request to read a optimization from a project.
optimizationId
string
Optimization ID for the optimization we want to read.
Example
{
"optimizationId": "string"
}
Responses
The /optimizations/read API provides a response in the following format:
200 Success
ReadOptimizationResponse Model - Response received when reading an optimization.
optimization
Optimization object
Response received when launching an optimization job.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"optimization": {
"optimizationId": "string",
"snapshotId": "string",
"projectId": "string",
"name": "string",
"status": "New",
"nodeType": "O2-8",
"parallelNodes": 4,
"criterion": {
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"extremum": "max or min",
"targetValue": 1
},
"runtimeStatistics": "string",
"constraints": [
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"operator": "greater",
"targetValue": 1
}
],
"parameters": [
{
"name": "rsi_period",
"min": 10,
"max": 20,
"step": 1,
"minStep": 1
}
],
"backtests": ,
"strategy":
"QuantConnect.Optimizer.Strategies.GridSearchOptimizationStrategy"requested": "2021-11-26T15:18:27.693Z",
"optimizationTarget": "string",
"gridLayout": [
"object"
],
"outOfSampleMaxEndDate": "string",
"outOfSampleDays": 0
},
"success": true,
"errors": [
"string"
]
}
Optimization Model - Response received when launching an optimization job.
optimizationId
string
Optimization ID.
snapshotId
string
Snapshot iD of this optimization.
projectId
string
Project ID of the project the optimization belongs to.
name
string
Name of the optimization.
status
string Enum
Status of the optimization. Options : ['New', 'Aborted',
'Running', 'Completed']
nodeType
string Enum
example: O2-8
Optimization node type. Options : ['O2-8', 'O4-12', 'O8-16']
parallelNodes
integer
example: 4
Number of parallel nodes for optimization.
criterion
OptimizationTarget object
/.
runtimeStatistics
string object
Dictionary representing a runtime banner/updating statistics
for the optimization.
constraints
OptimizationConstraint Array
Optimization constraints.
parameters
OptimizationParameter Array
Optimization parameters.
backtests
object
Dictionary of optimization backtests.
strategy
string
example:
QuantConnect.Optimizer.Strategies.GridSearchOptimizationStrategy
Optimization strategy.
requested
string($date-time)
Optimization requested date and time.
optimizationTarget
string
Statistic to be optimized.
gridLayout
object Array
List with grid charts representing the grid layout.
outOfSampleMaxEndDate
string
End date of out of sample data.
outOfSampleDays
integer
Number of days of out of sample days.
E
x
a
m
ple
{
"
o
p
t
i
m
i
z
a
t
i
o
n
I
d
": "
s
t
r
i
n
g
"
,
"
s
n
a
p
s
h
o
t
I
d
": "
s
t
r
i
n
g
"
,
"
p
r
o
j
e
c
t
I
d
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
s
t
a
t
u
s
": "
N
e
w
"
,
"
n
o
d
e
T
y
p
e
": "
O
2
-
8
"
,
"
p
a
r
a
l
l
e
l
N
o
d
e
s
": 4
,
"
c
r
i
t
e
r
i
o
n
": {
"
t
a
r
g
e
t
": "TotalPerform
a
n
c
e.P
o
r
t
f
o
l
i
o
S
t
a
t
i
s
t
i
c
s.S
h
a
r
p
e
R
a
t
i
o
"
,
"
e
x
t
r
e
m
u
m
": "
m
a
x
o
r
m
i
n
"
,
"
t
a
r
g
e
t
V
a
l
u
e
": 1
}
,
"
r
u
n
t
i
m
e
S
t
a
t
i
s
t
i
c
s
": "
s
t
r
i
n
g
"
,
"
c
o
n
s
t
r
a
i
n
t
s
": [
{
"
t
a
r
g
e
t
": "TotalPerforman
c
e.P
o
r
t
f
o
l
i
o
S
t
a
t
i
s
t
i
c
s.S
h
a
r
p
e
R
a
t
i
o
"
,
"
o
p
e
r
a
t
o
r
": "
g
r
e
a
t
e
r
"
,
"
t
a
r
g
e
t
V
a
l
u
e
": 1
}
]
,
"
p
a
r
a
m
e
t
e
r
s
": [
{
"
n
a
m
e
": "
r
s
i
_
p
e
r
i
o
d
"
,
"
m
i
n
": 1
0
,
"
m
a
x
": 2
0
,
"
s
t
e
p
": 1
,
"
m
i
n
S
t
e
p
": 1
}
]
,
"
b
a
c
k
t
e
s
t
s
": ,
"
s
t
r
a
t
e
g
y
": "QuantConnect.O
p
t
i
m
i
z
e
r.S
t
r
a
t
e
g
i
e
s.G
r
i
d
S
e
a
r
c
h
O
p
t
i
m
i
z
a
t
i
o
n
S
t
r
a
t
e
gy"
r
e
q
u
e
s
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
o
p
t
i
m
i
z
a
t
i
o
n
T
a
r
g
e
t
": "
s
t
r
i
n
g
"
,
"
g
r
i
d
L
a
y
o
u
t
": [
"
o
b
j
e
c
t
"
]
,
"
o
u
t
O
f
S
a
m
p
l
e
M
a
x
E
n
d
D
a
t
e
": "
s
t
r
i
n
g
"
,
"
o
u
t
O
f
S
a
m
p
l
e
D
a
y
s
": 0
}
OptimizationTarget Model
target
string
example: TotalPerformance.PortfolioStatistics.SharpeRatio
Property we want to track.
extremum
string
example: max or
min
Defines the direction of optimization.
targetValue
float
example: 1
The value of the property we want to track.
Example
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"extremum": "max or min",
"targetValue": 1
}
OptimizationConstraint Model
target
string
example: TotalPerformance.PortfolioStatistics.SharpeRatio
Property we want to track.
operator
string
example:
greater
The target comparison operation.
targetValue
float
example: 1
The value of the property we want to track.
Example
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"operator": "greater",
"targetValue": 1
}
OptimizationParameter Model
name
string
example: rsi_period
Name of optimization parameter.
min
float
example:
10
Minimum value of optimization parameter, applicable for
boundary conditions.
max
float
example:
20
Maximum value of optimization parameter, applicable for
boundary conditions.
step
float
example: 1
Movement, should be positive.
minStep
float
example: 1
Minimal possible movement for current parameter, should
be positive. Used by
Strategies.EulerSearchOptimizationStrategy to
determine when this parameter can no longer be optimized.
Example
{
"name": "rsi_period",
"min": 10,
"max": 20,
"step": 1,
"minStep": 1
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Optimization Management > Delete Optimization
Optimization Management
Delete Optimization
Introduction
Delete an optimization.
Request
The /optimizations/delete API accepts requests in the following format:
DeleteOptimizationRequest Model - Delete an optimization.
optimizationId
string
Optimization ID we want to delete.
Example
{
"optimizationId": "string"
}
Responses
The /optimizations/delete API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Optimization Management > Abort Optimization
Optimization Management
Abort Optimization
Introduction
Abort an optimization.
Request
The /optimizations/abort API accepts requests in the following format:
AbortOptimizationRequest Model - Abort an optimization.
optimizationId
string
Optimization ID we want to abort.
Example
{
"optimizationId": "string"
}
Responses
The /optimizations/abort API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Optimization Management > List Optimization
Optimization Management
List Optimization
Introduction
List all the optimizations for a project.
Request
Project ID we'd like to get a list of optimizations for. The /optimizations/list API accepts requests in the following format:
ListOptimizationRequest Model - Project ID we'd like to get a list of optimizations for.
projectId
integer
Project ID we'd like to get a list of optimizations for.
Example
{
"projectId": 0
}
Responses
The /optimizations/list API provides a response in the following format:
200 Success
ListOptimizationResponse Model - Response received when listing optimizations of a project.
optimizations
CreateOptimizationResponse Array
Collection of summarized optimization objects.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"optimizations": [
{
"optimizationId": "string",
"projectId": "string",
"name": "string",
"status": "New",
"nodeType": "O2-8",
"criterion": {
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"extremum": "max or min",
"targetValue": 1
},
"created": "2021-11-26T15:18:27.693Z",
"psr": 0,
"sharpeRatio": 0,
"trades": 0,
"cloneId": 0,
"outOfSampleDays": 0,
"outOfSampleMaxEndDate": "2021-11-
26T15:18:27.693Z",
"parameters": [
"object"
]
}
],
"success": true,
"errors": [
"string"
]
}
CreateOptimizationResponse Model - Response received when launching an optimization job.
optimizationId
string
Optimization ID.
projectId
string
Project ID of the project the optimization belongs to.
name
string
Name of the optimization.
status
string Enum
Status of the optimization. Options : ['New', 'Aborted',
'Running', 'Completed']
nodeType
string Enum
example: O2-8
Optimization node type. Options : ['O2-8', 'O4-12', 'O8-16']
criterion
OptimizationTarget object
/.
created
string($date-time)
Date when this optimization was created.
psr
number
Price-sales ratio stastic.
sharpeRatio
number
Sharpe ratio statistic.
trades
integer
Number of trades.
cloneId
integer
ID of project, were this current project was originally
cloned.
outOfSampleDays
integer
Number of days of out of sample days.
outOfSampleMaxEndDate
string($date-time)
End date of out of sample data.
parameters
object Array
Parameters used in this optimization.
E
x
a
m
ple
{
"
o
p
t
i
m
i
z
a
t
i
o
n
I
d
": "
s
t
r
i
n
g
"
,
"
p
r
o
j
e
c
t
I
d
": "
s
t
r
i
n
g
"
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
s
t
a
t
u
s
": "
N
e
w
"
,
"
n
o
d
e
T
y
p
e
": "
O
2
-
8
"
,
"
c
r
i
t
e
r
i
o
n
": {
"
t
a
r
g
e
t
": "TotalPerform
a
n
c
e.P
o
r
t
f
o
l
i
o
S
t
a
t
i
s
t
i
c
s.S
h
a
r
p
e
R
a
t
i
o
"
,
"
e
x
t
r
e
m
u
m
": "
m
a
x
o
r
m
i
n
"
,
"
t
a
r
g
e
t
V
a
l
u
e
": 1
}
,
"
c
r
e
a
t
e
d
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
p
s
r
": 0
,
"
s
h
a
r
p
e
R
a
t
i
o
": 0
,
"
t
r
a
d
e
s
": 0
,
"
c
l
o
n
e
I
d
": 0
,
"
o
u
t
O
f
S
a
m
p
l
e
D
a
y
s
": 0
,
"
o
u
t
O
f
S
a
m
p
l
e
M
a
x
E
n
d
D
a
t
e
": "
2
0
2
1
-
1
1
-
2
6
T
1
5:1
8:2
7.6
9
3
Z
"
,
"
p
a
r
a
m
e
t
e
r
s
": [
"
o
b
j
e
c
t
"
]
}
OptimizationTarget Model
target
string
example: TotalPerformance.PortfolioStatistics.SharpeRatio
Property we want to track.
extremum
string
example: max or
min
Defines the direction of optimization.
targetValue
float
example: 1
The value of the property we want to track.
Example
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"extremum": "max or min",
"targetValue": 1
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Optimization Management > Estimate Optimization Cost
Optimization Management
Estimate Optimization Cost
Introduction
Estimate the cost of an optimization with the specified parameters.
Request
Project, compile and optimization parameters for estimating the cost of an optimization job. The /optimizations/estimate API
accepts requests in the following format:
EstimateOptimizationRequest Model - Request to estimate the cost of an optimization job.
projectId
integer
Project ID of the project the optimization belongs to.
name
string
Name of the optimization.
target
string
Target of the optimization.
targetTo
string
example: max or
min
Target extremum of the optimization.
targetValue
float
Optimization target value.
strategy
string
example:
QuantConnect.Optimizer.Strategies.GridSearchOptimizationStrategy
Optimization strategy.
compileId
string
Optimization compile ID.
parameters
OptimizationParameter Array
Optimization parameters.
constraints
OptimizationConstraint Array
Optimization constraints.
E
x
a
m
ple
{
"
p
r
o
j
e
c
t
I
d
": 0
,
"
n
a
m
e
": "
s
t
r
i
n
g
"
,
"
t
a
r
g
e
t
": "
s
t
r
i
n
g
"
,
"
t
a
r
g
e
t
T
o
": "
m
a
x
o
r
m
i
n
"
,
"
t
a
r
g
e
t
V
a
l
u
e
": ,
"
s
t
r
a
t
e
g
y
": "QuantConnect.O
p
t
i
m
i
z
e
r.S
t
r
a
t
e
g
i
e
s.G
r
i
d
S
e
a
r
c
h
O
p
t
i
m
i
z
a
t
i
o
n
S
t
r
a
t
e
gy"
c
o
m
p
i
l
e
I
d
": "
s
t
r
i
n
g
"
,
"
p
a
r
a
m
e
t
e
r
s
": [
{
"
n
a
m
e
": "
r
s
i
_
p
e
r
i
o
d
"
,
"
m
i
n
": 1
0
,
"
m
a
x
": 2
0
,
"
s
t
e
p
": 1
,
"
m
i
n
S
t
e
p
": 1
}
]
,
"
c
o
n
s
t
r
a
i
n
t
s
": [
{
"
t
a
r
g
e
t
": "TotalPerforman
c
e.P
o
r
t
f
o
l
i
o
S
t
a
t
i
s
t
i
c
s.S
h
a
r
p
e
R
a
t
i
o
"
,
"
o
p
e
r
a
t
o
r
": "
g
r
e
a
t
e
r
"
,
"
t
a
r
g
e
t
V
a
l
u
e
": 1
}
]
}
OptimizationParameter Model
name
string
example: rsi_period
Name of optimization parameter.
min
float
example:
10
Minimum value of optimization parameter, applicable for
boundary conditions.
max
float
example:
20
Maximum value of optimization parameter, applicable for
boundary conditions.
step
float
example: 1
Movement, should be positive.
minStep
float
example: 1
Minimal possible movement for current parameter, should
be positive. Used by
Strategies.EulerSearchOptimizationStrategy to
determine when this parameter can no longer be optimized.
Example
{
"name": "rsi_period",
"min": 10,
"max": 20,
"step": 1,
"minStep": 1
}
OptimizationConstraint Model
target
string
example: TotalPerformance.PortfolioStatistics.SharpeRatio
Property we want to track.
operator
string
example:
greater
The target comparison operation.
targetValue
float
example: 1
The value of the property we want to track.
Example
{
"target":
"TotalPerformance.PortfolioStatistics.SharpeRatio",
"operator": "greater",
"targetValue": 1
}
Responses
The /optimizations/estimate API provides a response in the following format:
200 Success
EstimateOptimizationResponse Model - Response received when estimating the cost of an optimization.
estimate
Estimate object
Response received when estimating the cost of an
optimization.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"estimate": {
"estimateId": "string",
"time": 60,
"balance": 10
},
"success": true,
"errors": [
"string"
]
}
Estimate Model - Response received when estimating the cost of an optimization.
estimateId
string
Estimate Id.
time
integer
example:
60
Estimate time in seconds.
balance
integer
example:
10
Estimate balance in QCC.
Example
{
"estimateId": "string",
"time": 60,
"balance": 10
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Object Store Management
API Reference
Object Store Management
API Reference > Object Store Management > Upload Object Store Files
Object Store Management
Upload Object Store Files
Introduction
Upload files to the Object Store.
Request
Upload files to the Object Store. The /object/set API accepts requests in the following format:
The /object/set API requires a file request in the following format:
ObjectStoreBinaryFile Model - Represents a binary file we we'd like to upload the file to upload to the Object Store.
objectData
binary
Object data to be stored.
Example
{
"objectData": b"Hello, world!"
}
Responses
The /object/set API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Object Store Management > Get Object Store Metadata
Object Store Management
Get Object Store Metadata
Introduction
Get Object Store properties of a specific organization and key. It does not work if the object store is a directory.
Request
Get Object Store properties of a specific organization and key. The /object/properties API accepts requests in the following
format:
GetObjectStorePropertiesRequest Model - Request to get Object Store properties of a specific organization and keys.
organizationId
string
Organization ID we would like to get the Object Store
properties from.
key
string
Key to the Object Store.
Example
{
"organizationId": "string",
"key": "string"
}
Responses
The /object/properties API provides a response in the following format:
200 Success
GetObjectStorePropertiesResponse Model - Response received when fetching Object Store file properties.
metadata
ObjectStoreProperties object
Object Store file properties.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"metadata": {
"key": "string",
"modified": "2021-11-26T15:18:27.693Z",
"created": "string",
"size": 24,
"md5": "string",
"mime": "string",
"preview": "string"
},
"success": true,
"errors": [
"string"
]
}
ObjectStoreProperties Model - Object Store file properties.
key
string
Object Store key.
modified
string($date)
Last time it was modified.
created
string
Date this project was created.
size
float
example:
24
Object Store file size.
md5
string
MD5 (hashing algorithm) hash authentication code.
mime
string
MIME type.
preview
string
Preview of the Object Store file content.
Example
{
"key": "string",
"modified": "2021-11-26T15:18:27.693Z",
"created": "string",
"size": 24,
"md5": "string",
"mime": "string",
"preview": "string"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Object Store Management > Get Object Store File
Object Store Management
Get Object Store File
Introduction
Get Object Store file of a specific organization and key.
Request
Get Object Store files of a specific organization and key. The /object/get API accepts requests in the following format:
GetObjectStoreJobIdRequest Model - Request to get JobId for the requested Object Store files.
organizationId
string
Organization ID we would like to get the Object Store files
from.
keys
string Array
Keys to the Object Store files.
Example
{
"organizationId": "string",
"keys": [
"string"
]
}
GetObjectStoreURLRequest Model - Request to get a download URL for certain Object Store files.
organizationId
string
Organization ID we would like to get the Object Store files
from.
jobId
string
Job ID for getting a download URL for.
Example
{
"organizationId": "string",
"jobId": "string"
}
Responses
The /object/get API provides a response in the following format:
200 Success
GetObjectStoreResponse Model - Response received when fetching Object Store file.
jobId
string
Job ID which can be used for querying state or packaging.
url
string
The URL to download the object. This can also be null.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"jobId": "string",
"url": "string",
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Object Store Management > Delete Object Store File
Object Store Management
Delete Object Store File
Introduction
Delete the Object Store file of a specific organization and key.
Request
Delete the Object Store file of a specific organization and key. The /object/delete API accepts requests in the following
format:
DeleteObjectStoreRequest Model - Request to delete Object Store metadata of a specific organization and key.
organizationId
string
Organization ID we'd like to delete the Object Store file
from.
key
string
Key to the Object Store file.
Example
{
"organizationId": "string",
"key": "string"
}
Responses
The /object/delete API provides a response in the following format:
200 Success
RestResponse Model - Base API response class for the QuantConnect API.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Object Store Management > List Object Store Files
Object Store Management
List Object Store Files
Introduction
List the Object Store files of a specific organization and path.
Request
List the Object Store files of a specific organization and path. The /object/list API accepts requests in the following format:
ListObjectStoreRequest Model - Request to list Object Store files of a specific organization and path.
organizationId
string
Organization ID we'd like to list the Object Store files from.
path
string
Path to the Object Store files.
Example
{
"organizationId": "string",
"path": "string"
}
Responses
The /object/list API provides a response in the following format:
200 Success
ListObjectStoreResponse Model - Response received containing a list of stored objects metadata, as well as the total size
of all of them.
path
string
example: Mia
Path to the files in the Object Store.
objects
ObjectStoreSummary Array
List of objects stored.
objectStorageUsed
int
Size of all objects stored in bytes.
objectStorageUsedHuman
string
Size of all the objects stored in human-readable format.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"path": "Mia",
"objects": [
{
"key": "Mia/Test",
"name": "string",
"modified": "2021-11-26T15:18:27.693Z",
"mime": "application/json",
"folder": true,
"size": 13
}
],
"objectStorageUsed": ,
"objectStorageUsedHuman": "string",
"success": true,
"errors": [
"string"
]
}
ObjectStoreSummary Model - Summary information of the Object Store.
key
string
example: Mia/Test
Object Store key.
name
string
File or folder name.
modified
string($date-time)
Last time it was modified.
mime
string
example: application/json MIME type.
folder
boolean
True if it is a folder, false otherwise.
size
float
example:
13
Object Store file size.
Example
{
"key": "Mia/Test",
"name": "string",
"modified": "2021-11-26T15:18:27.693Z",
"mime": "application/json",
"folder": true,
"size": 13
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Reports
API Reference
Reports
The QuantConnect REST API lets you access your backtest reports from our cloud servers through URL endpoints.
Backtest Report
API Reference > Reports > Backtest Report
Reports
Backtest Report
Introduction
Read out the report of a backtest in the project Id specified.
Request
A JSON object containing info about the project to delete. The /backtests/read/report API accepts requests in the following
format:
BacktestReportRequest Model - Request to read out the report of a backtest.
projectId
integer
Id of the project to read.
backtestId
string
Specific backtest Id to read.
Example
{
"projectId": 0,
"backtestId": "string"
}
Responses
The /backtests/read/report API provides a response in the following format:
200 Success
BacktestReport Model - Backtest Report Response wrapper.
report
string
HTML data of the report with embedded base64 images.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"report": "string",
"success": true,
"errors": [
"string"
]
}
RequestFailedError Model - The API method call could not be completed as requested.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"success": true,
"errors": [
"string"
]
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Account
API Reference
Account
Introduction
None
Description
Will read the organization account status.
Request
The /account/read API accepts requests in the following format:
/account/read Method
/account/read method takes no parameters.
Responses
The /account/read API provides a response in the following format:
200 Success
AccountResponse Model - Account information for an organization.
organizationId
string
The organization Id.
creditBalance
number
The current account balance.
card
Card object
Credit card information.
Example
{
"organizationId": "string",
"creditBalance": 0,
"card": {
"brand": "string",
"expiration": "2021-11-26T15:18:27.693Z",
"last4": "string"
}
}
Card Model - Credit card information.
brand
string
Credit card brand.
expiration
string($date-time)
The credit card expiration.
last4
string
The last 4 digits of the card.
Example
{
"brand": "string",
"expiration": "2021-11-26T15:18:27.693Z",
"last4": "string"
}
401 Authentication Error
UnauthorizedError Model - Unauthorized response from the API. Key is missing, invalid, or timestamp is too old for hash.
www_authenticate
string
Header
API Reference > Lean Version
API Reference
Lean Version
Introduction
Returns a list of lean versions with basic information for each version.
Request
The /lean/versions/read API accepts requests in the following format:
/lean/versions/read Method
/lean/versions/read method takes no parameters.
Responses
The /lean/versions/read API provides a response in the following format:
200 Success
LeanVersionsResponse Model - Contains the LEAN versions with their basic descriptions.
versions
object Array
List of LEAN versions with their basic descriptions.
success
boolean
Indicate if the API request was successful.
errors
string Array
List of errors with the API call.
Example
{
"versions": [
"object"
],
"success": true,
"errors": [
"string"
]
}
API Reference > Examples
API Reference
Examples
Introduction
The following page is a collection of API implementation examples to showcase using the QuantConnect API. If you have an
implementation of the QuantConnect API, let us know so we can showcase your project.
Complete Test Implementation
This Python research notebook is an example that implements all of the QuantConnect API in vanilla Python to demonstrate the
JSON calls and responses.
Loading [MathJax]/jax/output/HTML-CSS/jax.js
Loading [MathJax]/jax/output/HTML-CSS/jax.js