# BenchMind AWS Lightsail deployment

This deployment runs Streamlit, FastAPI, and Caddy on one Linux instance. Caddy
terminates HTTPS, while FastAPI and Streamlit remain reachable only through the
Docker network.

## 1. Create the AWS resources

1. Create an AWS Budget alert before launching resources.
2. Create an Ubuntu 24.04 Lightsail instance. The 4 GB plan is recommended for
   the local embedding model and Chroma.
3. Attach a static IP.
4. Open TCP 80 and 443 to the internet. Restrict TCP 22 to your own IP.
5. Create DNS `A` records for the app and API subdomains pointing to the static
   IP.

## 2. Prepare the instance

Connect over SSH, then install Docker:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and reconnect so the Docker group change takes effect.

## 3. Give the instance read-only GitHub access

Create a dedicated deploy key on the Lightsail instance. Do not reuse your
personal SSH key:

```bash
ssh-keygen -t ed25519 -C "benchmind-lightsail" -f ~/.ssh/benchmind_deploy -N ""
cat ~/.ssh/benchmind_deploy.pub
```

In the private GitHub repository, open **Settings > Deploy keys > Add deploy
key**, paste the public key, and leave **Allow write access** disabled. Then
configure SSH and clone the repository:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github-benchmind
  HostName github.com
  User git
  IdentityFile ~/.ssh/benchmind_deploy
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config ~/.ssh/benchmind_deploy
cat >> ~/.ssh/known_hosts <<'EOF'
github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl
EOF
chmod 600 ~/.ssh/known_hosts
sudo mkdir -p /opt/benchmind
sudo chown "$USER":"$USER" /opt/benchmind
git clone git@github-benchmind:OWNER/REPOSITORY.git /opt/benchmind
cd /opt/benchmind
```

Replace `OWNER/REPOSITORY` with the actual GitHub path. The pinned Ed25519 host
key above must match the fingerprint published in GitHub's documentation.

## 4. Configure the production environment

Create `.env` on the server; never commit it:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=replace_me
GEMINI_MODEL=gemini-3.5-flash-lite

BENCHMIND_API_TOKEN=replace_with_output_of_openssl_rand_hex_32
APP_DOMAIN=benchmind.example.com
API_DOMAIN=api.benchmind.example.com
APP_AUTH_USER=benchmind
APP_AUTH_PASSWORD_HASH='$2a$14$replace_with_caddy_hash'
```

Generate the token with:

```bash
openssl rand -hex 32
```

Generate the browser login password hash without putting the plaintext password
in shell history:

```bash
read -rsp "App password: " APP_PASSWORD && echo
docker run --rm caddy:2.10-alpine caddy hash-password --plaintext "$APP_PASSWORD"
unset APP_PASSWORD
```

Store the resulting hash in `.env` inside single quotes so Docker Compose does
not interpret the dollar signs. The plaintext password must not be stored in
the project or `.env`.

The API documentation will be available at `https://API_DOMAIN/docs`. Protected
endpoints require `Authorization: Bearer <BENCHMIND_API_TOKEN>`. The token stays
inside the server-side Streamlit container and is not sent to the browser.

## 5. Deploy and verify

```bash
mkdir -p .benchmind
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 api web proxy
curl --fail "https://$APP_DOMAIN/_stcore/health"
curl --fail "https://$API_DOMAIN/health"
```

Caddy obtains and renews TLS certificates automatically after DNS points to the
instance and ports 80/443 are reachable.

## 6. Connect GitHub Actions to Lightsail

Install a GitHub Actions self-hosted runner from the private repository's
**Settings > Actions > Runners > New self-hosted runner** page. Run the commands
shown there as the same non-root Linux user that owns `/opt/benchmind` and can
run Docker. Install and start it as a service using the generated `svc.sh`
commands. The repository-level runner's default Linux and x64 labels are used.

The workflow in `.github/workflows/deploy.yml` first runs the test suite on a
GitHub-hosted runner. If tests pass, the Lightsail runner executes
`deploy/deploy.sh`, which pulls `main` with a fast-forward-only merge, rebuilds
the containers, waits for their health checks, and prints their status.

This runner makes an outbound HTTPS connection to GitHub, so port 22 can remain
restricted to your own IP. Protect the `main` branch and limit write access:
any workflow committed to a branch that can target this runner can execute as
the runner's Linux user.

## 7. Update or roll back

Push or merge into `main` to deploy automatically. Before a high-risk update,
create a Lightsail snapshot. A manual deployment can be run with:

```bash
bash deploy/deploy.sh
```

Inspect failures with `docker compose logs --tail=200 <service>`. To roll back,
revert the problematic commit on GitHub and push the revert; do not rewrite the
shared `main` branch. Restore the instance snapshot if an application or data
migration cannot be recovered.
