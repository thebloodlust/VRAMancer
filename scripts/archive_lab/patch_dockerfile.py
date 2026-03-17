import sys
with open("Dockerfile", "r") as f:
    df = f.read()

df = df.replace("EXPOSE 5030", "EXPOSE 5030\nEXPOSE 8081\nEXPOSE 9108")
with open("Dockerfile", "w") as f:
    f.write(df)

