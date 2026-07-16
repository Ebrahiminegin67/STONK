import kagglehub

# Download latest version
path = kagglehub.dataset_download("ocanaydin/italian-telecom-data-2013-1week")

print("Path to dataset files:", path)
