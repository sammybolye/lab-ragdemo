import requests
import re

url = "http://localhost:8888/login"
password = "demo"

s = requests.Session()

# 1. Get the login page to get the cookie and xsrf
r = s.get(url)
print(f"Initial Page: {r.status_code}")

# Extract _xsrf from cookie
xsrf = s.cookies.get("_xsrf")
if not xsrf:
    # Sometimes it's in the body or header, but usually cookie for requests
    match = re.search(r'name="_xsrf" value="(.*?)"', r.text)
    if match:
        xsrf = match.group(1)

if not xsrf:
    print("FAILED: Could not find XSRF token")
    exit(1)

print(f"Found XSRF: {xsrf[:5]}...")

# 2. Post Password
payload = {
    "password": password,
    "_xsrf": xsrf
}

# Jupyter redirects after login. allow_redirects=True to follow it.
r = s.post(url, data=payload, allow_redirects=True)

# 3. Validation
# If successful, we should be at /lab or /tree, and NOT /login
print(f"Login Response: {r.status_code}")
print(f"Final URL: {r.url}")

if "login" not in r.url and r.status_code == 200:
    print("SUCCESS: Validated password 'demo' works.")
else:
    print("FAILURE: Could not log in. Still on login page.")
    exit(1)
