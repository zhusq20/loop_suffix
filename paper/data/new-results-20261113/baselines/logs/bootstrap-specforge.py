from pathlib import Path
import subprocess,os,json,time
base=Path("/ossfs/workspace/sz/loopfill/experiments/baselines")
report={"started_unix":time.time(),"status":"running"}
commands=[
 ["/tmp/loopfill-dflash-venv/bin/python","-m","pip","install","uv"],
 ["/tmp/loopfill-dflash-venv/bin/uv","python","install","3.11"],
 ["/tmp/loopfill-dflash-venv/bin/uv","venv","--python","3.11","/tmp/loopfill-specforge-venv"],
 ["/tmp/loopfill-dflash-venv/bin/uv","pip","install","--python","/tmp/loopfill-specforge-venv/bin/python","-e",str(base/"third_party/SpecForge")],
]
try:
 env=dict(os.environ,UV_PYTHON_INSTALL_DIR="/tmp/loopfill-python",UV_CACHE_DIR="/tmp/loopfill-uv-cache")
 for cmd in commands:
  print("COMMAND",cmd,flush=True)
  subprocess.run(cmd,env=env,check=True)
 report["status"]="installed_not_validated"
except Exception as exc:
 report["status"]="failed"
 report["error"]=repr(exc)
finally:
 report["finished_unix"]=time.time()
 report["commands"]=commands
 (base/"configs/specforge-environment-install.json").write_text(json.dumps(report,indent=2))
