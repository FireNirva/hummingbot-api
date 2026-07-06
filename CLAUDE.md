# CLAUDE.md

> ⚠ **休眠子系统(2026-07-06 戳记):** 本 repo 2025 末后基本停更;下文信息按当时快照阅读,现值(meme-sniper 活跃部署/配置)以 `deploy/meme-sniper/CLAUDE.md` 与工作区根 CLAUDE.md 为准。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Hummingbot API?

A FastAPI-based central hub for managing Hummingbot trading bots across multiple exchanges (CEX and DEX). The API orchestrates bot deployment, portfolio tracking, trade execution, and market data feeds through a microservice architecture using Docker containers, MQTT messaging, and PostgreSQL.

## Common Development Commands

### Environment Setup

```bash
# First-time setup (creates services and environment)
chmod +x setup.sh
./setup.sh

# Install development environment (requires Conda)
make install

# Install pre-commit hooks (automatically runs formatters)
make install-pre-commit

# Uninstall environment
make uninstall
```

### Running the API

```bash
# Production mode (Docker container)
./run.sh

# Development mode (source with hot-reload)
./run.sh --dev
# This starts PostgreSQL and EMQX in Docker, then runs FastAPI from source

# Manual development run (after activating conda env)
conda activate hummingbot-api
uvicorn main:app --reload
```

### Docker Operations

```bash
# Build API image
make build

# Deploy all services (API, PostgreSQL, EMQX)
make deploy

# View running containers
docker ps

# View logs
docker logs hummingbot-api
docker logs hummingbot-postgres
docker logs hummingbot-broker

# Stop all services
docker compose down

# Stop and remove volumes (⚠️ deletes all data)
docker compose down -v
```

### Code Quality

```bash
# Format code (runs automatically via pre-commit)
black --line-length 130 .
isort --line-length 130 --profile black .

# Pre-commit hooks check:
# - Private key detection
# - Wallet private key detection
# - isort (line-length 130, black profile)
# - flake8 (max line-length 130)
```

### Database Operations

```bash
# Fix database issues (automated repair script)
chmod +x fix-database.sh
./fix-database.sh

# Manual database access (⚠️ use 'hbot' user, NOT 'postgres')
docker exec -it hummingbot-postgres psql -U hbot -d hummingbot_api

# View database logs
docker logs hummingbot-postgres
```

### API Access

```bash
# Swagger UI (interactive API docs)
open http://localhost:8000/docs

# Test root endpoint
curl -u admin:admin http://localhost:8000/

# ReDoc (alternative docs)
open http://localhost:8000/redoc
```

## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
│                    (main.py, port 8000)                  │
│                                                           │
│  Lifespan Manager:                                       │
│  - Initializes services on startup                       │
│  - Manages graceful shutdown                             │
│  - Cleans up resources                                   │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌───────────────┐
│BotsOrchestrator│ │AccountsService │  │ DockerService │
│               │  │                │  │               │
│ - Lifecycle   │  │ - Portfolio    │  │ - Containers  │
│ - MQTT comm   │  │ - Balances     │  │ - Images      │
│ - Monitoring  │  │ - Credentials  │  │ - Archiving   │
└───────────────┘  └────────────────┘  └───────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌───────────────┐
│ MQTTManager   │  │  PostgreSQL    │  │ Docker Engine │
│ (EMQX broker) │  │  (Database)    │  │               │
└───────────────┘  └────────────────┘  └───────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│   Bot Containers (Hummingbot)         │
│   - Isolated execution                │
│   - MQTT communication                │
│   - File-based state                  │
└───────────────────────────────────────┘
```

### Core Services (`services/`)

**1. BotsOrchestrator** (`bots_orchestrator.py`)
- Manages bot container lifecycle (start, stop, monitor)
- Handles MQTT communication with bot instances
- Tracks active bots from Docker and MQTT discovery
- Maintains `active_bots` dictionary and `stopping_bots` set

**2. AccountsService** (`accounts_service.py`)
- Manages exchange credentials (encrypted storage)
- Portfolio tracking across all accounts
- Periodic balance updates (`ACCOUNT_UPDATE_INTERVAL` minutes)
- Aggregates CEX and DEX connector data

**3. DockerService** (`docker_service.py`)
- Docker SDK wrapper for container operations
- Async image pulling with progress tracking
- Container lifecycle and health monitoring
- Archives bot data to local filesystem or S3

**4. GatewayService** (`gateway_service.py`)
- Manages Gateway container for DEX trading
- Gateway lifecycle and configuration
- Blockchain network interfaces (Solana, Ethereum, etc.)
- Uses `GatewayClient` for API communication

**5. MarketDataFeedManager** (`market_data_feed_manager.py`)
- Real-time market data feeds (candles, prices, orderbooks)
- Feed lifecycle with auto-cleanup after inactivity
- Uses non-trading connector instances (data-only)
- Background cleanup task runs every `MARKET_DATA_CLEANUP_INTERVAL` seconds

### Data Layer (Repository Pattern)

**Database Models** (`database/models.py`)
- SQLAlchemy async models for PostgreSQL
- Tables:
  - `orders`: Order history with exchange info
  - `trades`: Executed trades with fees
  - `account_balances`: Periodic balance snapshots
  - `positions`: Perpetual contract positions
  - `funding_payments`: Funding payment history
  - `gateway_swaps`: DEX swap transactions
  - `gateway_clmm_positions`: Concentrated liquidity positions
  - `bot_runs`: Bot execution history and metadata

**Repositories** (`database/repositories/`)
Each repository uses async database sessions:
- `AccountRepository`: Balance and account state queries
- `OrderRepository`: Order history with filtering and pagination
- `TradeRepository`: Trade execution records
- `FundingRepository`: Perpetual funding payments
- `GatewaySwapRepository`: DEX swap transactions
- `GatewayClmmRepository`: CLMM position management
- `BotRunRepository`: Bot execution analytics

**Database Connection** (`database/connection.py`)
```python
from database.connection import get_db_session

async def my_function():
    async with get_db_session() as session:
        result = await repository.get_something(session)
```

### Router Layer (`routers/`)

All routers use dependency injection to access services from `request.app.state`:

- **`accounts.py`**: Account and credential management
- **`bot_orchestration.py`**: Bot deployment and monitoring
- **`connectors.py`**: Exchange connector discovery and config
- **`controllers.py`**: V2 strategy controller CRUD operations
- **`scripts.py`**: V1/V2 script management
- **`trading.py`**: Order placement, positions, trades
- **`portfolio.py`**: Portfolio state and analytics
- **`market_data.py`**: Real-time and historical market data
- **`gateway.py`**: Gateway lifecycle and configuration
- **`gateway_swap.py`**: DEX swap operations (Jupiter, 0x, etc.)
- **`gateway_clmm.py`**: Concentrated liquidity positions
- **`docker.py`**: Docker operations and image management
- **`backtesting.py`**: Strategy backtesting engine
- **`archived_bots.py`**: Stopped bot analytics

### Bot File Structure

Each bot maintains isolated state:

```
bots/
├── instances/
│   └── hummingbot-{bot_name}/
│       ├── conf/                    # Configuration files
│       │   ├── controllers/         # V2 controller configs
│       │   └── conf_client.yml      # Bot settings
│       ├── data/                    # SQLite database (bot state)
│       └── logs/                    # Execution logs
├── controllers/                     # V2 controller templates
│   ├── market_making/              # PMM strategies
│   ├── directional_trading/        # Trend-following
│   └── generic/                    # Multi-purpose
├── scripts/                        # V1/V2 script templates
└── credentials/                    # Encrypted credentials
    └── master_account/
        ├── conf_client.yml
        ├── conf_fee_overrides.yml
        └── .password_verification  # Password verification file
```

**Important**: The API never directly modifies bot instance files. All communication happens through MQTT commands.

### Configuration System (`config.py`)

**Pydantic Settings with nested configuration classes:**

```python
from config import settings

# Access settings
settings.security.username
settings.broker.host
settings.database.url
settings.market_data.cleanup_interval
```

**Environment variable prefixes:**
- `BROKER_*`: MQTT broker settings (host, port, username, password)
- `DATABASE_*`: Database URL
- `MARKET_DATA_*`: Feed cleanup and timeout intervals
- `GATEWAY_*`: Gateway service URL
- `AWS_*`: S3 archiving credentials
- No prefix: Security settings (USERNAME, PASSWORD, CONFIG_PASSWORD, DEBUG_MODE)

**Key settings:**
- `ACCOUNT_UPDATE_INTERVAL`: Balance update frequency (minutes, default: 5)
- `MARKET_DATA_CLEANUP_INTERVAL`: Feed cleanup interval (seconds, default: 300)
- `MARKET_DATA_FEED_TIMEOUT`: Feed inactivity timeout (seconds, default: 600)
- `BANNED_TOKENS`: Comma-separated list of excluded tokens

## Key Development Concepts

### 1. MQTT Communication Pattern

Bots publish state updates and subscribe to commands via MQTT:

```python
# Topic structure:
# Bot -> API: hbot/{bot_name}/status
# API -> Bot: hbot/{bot_name}/command
```

**MQTTManager** (`utils/mqtt_manager.py`):
- Auto-discovery via MQTT messages (30-second timeout)
- Message caching for latest bot states
- Topic subscription management
- Connection resilience with reconnection logic
- Uses `aiomqtt` for async communication

**Bot Discovery:**
```python
# BotsOrchestrator combines two sources:
docker_bots = await self.get_active_containers()
mqtt_bots = self.mqtt_manager.get_discovered_bots(timeout_seconds=30)
all_active_bots = set(docker_bots + mqtt_bots)
```

### 2. Bot Deployment Flow

**V2 Controllers** (recommended):
1. Create controller config: `POST /controllers/configs/{config_name}`
2. Deploy bot: `POST /bot-orchestration/deploy-v2-controllers`
3. API creates Docker container with mounted volumes
4. Bot connects to MQTT and publishes status
5. BotsOrchestrator tracks bot state

**V2 Scripts** (legacy):
1. Create script config: `POST /scripts/configs/{config_name}`
2. Deploy bot: `POST /bot-orchestration/deploy-v2-script`

**Container creation** (`DockerService`):
```python
# Environment variables passed to bot container:
- CONFIG_PASSWORD
- BROKER_HOST, BROKER_PORT
- Connector-specific credentials (from encrypted storage)
```

### 3. Authentication & Security

**HTTP Basic Auth** (`main.py:auth_user`):
- All endpoints require authentication (except in DEBUG_MODE)
- Uses `secrets.compare_digest` for constant-time comparison
- Credentials from environment variables

**Credential Encryption** (`utils/security.py`, `BackendAPISecurity`):
- Exchange API keys encrypted with CONFIG_PASSWORD
- Uses Hummingbot's `ETHKeyFileSecretManger` (Fernet encryption)
- Password verification file: `bots/credentials/master_account/.password_verification`
- Auto-creates verification file on first startup if missing

### 4. Database Access Patterns

**Async-first design:**
```python
from database.connection import get_db_session
from sqlalchemy import select

async def get_orders(session, account_name: str):
    stmt = select(Order).where(Order.account_name == account_name)
    result = await session.execute(stmt)
    return result.scalars().all()

# Usage:
async with get_db_session() as session:
    orders = await get_orders(session, "my_account")
```

**Pagination strategies:**
- **Cursor-based** (preferred for large datasets):
  ```python
  # Returns {"data": [...], "next_cursor": "..."}
  # Client sends cursor in next request
  ```
- **Offset-based** (simple cases):
  ```python
  # Traditional limit/offset pagination
  ```

### 5. Real-time Data Synchronization

**Account Balances:**
- Periodic updates every `ACCOUNT_UPDATE_INTERVAL` minutes
- Real-time updates from bot MQTT messages
- Historical snapshots stored in PostgreSQL
- `AccountsService` maintains update loop

**Market Data Feeds:**
- Auto-cleanup after `FEED_TIMEOUT` seconds of inactivity
- Each feed runs in background asyncio task
- Multiple subscribers share same feed (memory optimization)
- `MarketDataFeedManager` tracks active feeds

### 6. Gateway for DEX Trading

**Gateway Container:**
- Separate Docker container (hummingbot/gateway)
- Manages blockchain wallet connections
- Unified interface for DEX protocols (Jupiter, Meteora, 0x, etc.)
- Communicates via HTTP REST API

**Networking:**
- macOS/Windows: Uses `host.docker.internal`
- Linux: Requires `extra_hosts` in docker-compose.yml
- Default port: 15888
- Dev mode: HTTP, Production: HTTPS with certificates

**GatewayService operations:**
- Start/stop/restart container
- Add/remove tokens and networks
- Execute swaps and manage CLMM positions
- Uses `GatewayClient` for API communication

## Development Patterns

### Adding a New Router

1. Create router file in `routers/`:
```python
from fastapi import APIRouter, Depends, Request

router = APIRouter(prefix="/my-feature", tags=["My Feature"])

@router.get("/endpoint")
async def my_endpoint(request: Request):
    # Access services from app.state
    service = request.app.state.my_service
    return await service.do_something()
```

2. Include router in `main.py`:
```python
from routers import my_feature

# In main.py, after creating app:
app.include_router(my_feature.router, dependencies=[Depends(auth_user)])
```

### Adding a New Service

1. Create service in `services/my_service.py`:
```python
class MyService:
    def __init__(self, config_param):
        self.config_param = config_param
        self._task = None

    def start(self):
        """Called during app startup (sync or async)"""
        self._task = asyncio.create_task(self._background_loop())

    async def stop(self):
        """Called during app shutdown"""
        if self._task:
            self._task.cancel()
```

2. Initialize in `main.py` lifespan:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    my_service = MyService(settings.my_config)
    app.state.my_service = my_service
    my_service.start()

    yield

    # Shutdown
    await my_service.stop()
```

### Adding Database Models & Repositories

1. Add model in `database/models.py`:
```python
from sqlalchemy import Column, Integer, String, DateTime
from database.connection import Base

class MyModel(Base):
    __tablename__ = "my_table"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime)
```

2. Create repository in `database/repositories/my_repository.py`:
```python
from sqlalchemy import select
from database.models import MyModel

class MyRepository:
    @staticmethod
    async def get_by_id(session, id: int):
        stmt = select(MyModel).where(MyModel.id == id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def create(session, name: str):
        instance = MyModel(name=name)
        session.add(instance)
        await session.commit()
        return instance
```

3. Export in `database/repositories/__init__.py`:
```python
from .my_repository import MyRepository
```

### Working with Bot Controllers

**Controller location:** `bots/controllers/{controller_type}/{controller_name}.py`

**Controller types:**
- `market_making/`: Market-making strategies (PMM, DMAN)
- `directional_trading/`: Trend-following (MACD, Bollinger, RSI)
- `generic/`: Multi-purpose strategies

**Controller structure:**
```python
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase

class MyController(ControllerBase):
    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize from config

    async def update_processed_data(self):
        # Process market data
        pass

    def process_update(self):
        # Generate orders based on processed data
        pass
```

**Controller config:**
- Stored in `bots/instances/{bot_name}/conf/controllers/{config_name}.json`
- Can have bot-specific overrides
- Validated against controller's Pydantic schema

## Common Pitfalls & Solutions

### 1. Database Connection Issues

**Problem:** "role 'postgres' does not exist"
**Cause:** PostgreSQL container creates only 'hbot' user, not default 'postgres' superuser
**Solution:** Always use `-U hbot` when connecting:
```bash
docker exec -it hummingbot-postgres psql -U hbot -d hummingbot_api
```

**Problem:** "database 'hummingbot_api' does not exist"
**Solution:** Run automated fix:
```bash
./fix-database.sh
```

### 2. MQTT Connection Failures

**Problem:** Bots not appearing in active list
**Cause:** MQTT broker not running or connection issues
**Check:**
```bash
docker logs hummingbot-broker
docker compose restart emqx

# Access EMQX dashboard (optional)
open http://localhost:18083
# Default: admin/public
```

### 3. Gateway Container Conflicts

**Problem:** Multiple Gateway containers causing issues
**Solution:** Keep only one:
```bash
docker ps -a | grep gateway
docker stop old-gateway-name
docker rm old-gateway-name

# Gateway container should be named 'gateway'
docker rename current-name gateway
```

### 4. Bot State Not Updating

**Problem:** Bot state stale in API
**Cause:** MQTT connection lost or bot crashed
**Check:**
```bash
# Check MQTT connection status
GET /bot-orchestration/mqtt

# Check specific bot status
GET /bot-orchestration/{bot_name}/status

# View bot container logs
docker logs hummingbot-{bot_name}
```

### 5. Password Verification File Missing

**Problem:** `[Errno 2] No such file or directory: '.password_verification'`
**Cause:** First-time setup not completed
**Solution:**
```bash
./setup.sh  # Initializes master_account and creates verification file
```

### 6. Pre-commit Hook Failures

**Problem:** Commit blocked by pre-commit hooks
**Common causes:**
- Private key detected in code
- Code not formatted with black/isort
- Flake8 style violations

**Solution:**
```bash
# Format code automatically
black --line-length 130 .
isort --line-length 130 --profile black .

# Check for keys manually
grep -r "private.*key" --include="*.py"

# Bypass hooks (⚠️ not recommended)
git commit --no-verify
```

## Environment Variables Reference

**Security:**
- `USERNAME`: API basic auth username (default: admin)
- `PASSWORD`: API basic auth password (default: admin)
- `CONFIG_PASSWORD`: Encryption key for bot credentials (required)
- `DEBUG_MODE`: Disable auth for development (default: false)

**Services:**
- `BROKER_HOST`: MQTT broker hostname (default: localhost, Docker: emqx)
- `BROKER_PORT`: MQTT broker port (default: 1883)
- `BROKER_USERNAME`: MQTT auth username (default: admin)
- `BROKER_PASSWORD`: MQTT auth password (default: password)
- `DATABASE_URL`: PostgreSQL connection string
- `GATEWAY_URL`: Gateway service URL (default: http://localhost:15888)

**Application:**
- `ACCOUNT_UPDATE_INTERVAL`: Balance update frequency in minutes (default: 5)
- `MARKET_DATA_CLEANUP_INTERVAL`: Feed cleanup interval (seconds, default: 300)
- `MARKET_DATA_FEED_TIMEOUT`: Feed inactivity timeout (seconds, default: 600)
- `LOGFIRE_ENVIRONMENT`: Observability environment name (default: dev)
- `BANNED_TOKENS`: Comma-separated token list (default: NAV,ARS,ETHW,ETHF)

**Optional (AWS S3 Archiving):**
- `AWS_API_KEY`: S3 access key
- `AWS_SECRET_KEY`: S3 secret key
- `AWS_S3_DEFAULT_BUCKET_NAME`: Default bucket for bot archives

**Optional (Monitoring):**
- `LOGFIRE_TOKEN`: Pydantic Logfire integration token

## Code Style & Standards

**Line length:** 130 characters (black, isort, flake8)
**Import sorting:** isort with black profile
**Pre-commit hooks:** Auto-format on commit
**Async-first:** All database and I/O operations use async/await
**Type hints:** Encouraged for public APIs
**Docstrings:** Required for routers and public service methods

**Pre-commit checks:**
- `detect-private-key`: Prevents committing API keys
- `detect-wallet-private-key`: Blockchain wallet key detection
- `isort`: Import sorting (line-length 130, black profile)
- `flake8`: Linting (max-line-length 130)

## Using Hummingbot API with AI Assistants (MCP)

### MCP Server Setup

The Hummingbot MCP server provides natural language access to all API functionality.

**Claude Desktop:**
Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "hummingbot": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "-e", "HUMMINGBOT_API_URL=http://host.docker.internal:8000", "-v", "hummingbot_mcp:/root/.hummingbot_mcp", "hummingbot/hummingbot-mcp:latest"]
    }
  }
}
```

**Claude Code CLI:**
```bash
claude mcp add --transport stdio hummingbot -- docker run --rm -i -e HUMMINGBOT_API_URL=http://host.docker.internal:8000 -v hummingbot_mcp:/root/.hummingbot_mcp hummingbot/hummingbot-mcp:latest
```

### MCP Tools Available

**Essential:**
- `configure_api_servers`: Configure API connection (run first!)
- `get_portfolio_overview`: Unified portfolio view
- `setup_connector`: Progressive credential setup

**Trading:**
- `place_order`: Execute CEX orders
- `search_history`: Order/trade/position history
- `set_account_position_mode_and_leverage`: Perpetual config

**Market Data:**
- `get_prices`: Current prices
- `get_candles`: OHLCV data
- `get_funding_rate`: Perpetual funding
- `get_order_book`: Order book analysis

**Gateway (DEX):**
- `manage_gateway_container`: Gateway lifecycle
- `manage_gateway_config`: Networks, tokens, connectors
- `manage_gateway_swaps`: DEX trading
- `explore_gateway_clmm_pools`: Pool discovery
- `manage_gateway_clmm_positions`: LP position management

**Bots:**
- `explore_controllers`: Strategy discovery
- `modify_controllers`: Create/update strategies
- `deploy_bot_with_controllers`: Deploy bots
- `get_active_bots_status`: Monitor bots
- `manage_bot_execution`: Start/stop bots

**For complete MCP tool documentation, see API_REFERENCE.md**

## API Documentation

**Swagger UI:** http://localhost:8000/docs (interactive, recommended)
**ReDoc:** http://localhost:8000/redoc (alternative view)
**OpenAPI JSON:** http://localhost:8000/openapi.json (machine-readable)

**Authentication:** All endpoints require HTTP Basic Auth with configured credentials.

**For detailed endpoint documentation, see API_REFERENCE.md**
