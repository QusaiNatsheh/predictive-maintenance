#!/bin/sh
# Download the two public datasets from their official sources and verify them.
# NASA C-MAPSS: Saxena & Goebel (2008), NASA Prognostics Data Repository (no license named; "use at own risk").
# UCI AI4I 2020: CC BY 4.0, https://doi.org/10.24432/C5HS5C
set -e
DEST=${1:-data/raw}
mkdir -p "$DEST/cmapss" "$DEST/ai4i"
curl -sSL --fail -o "$DEST/cmapss/turbofan.zip" \
  "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
curl -sSL --fail -o "$DEST/ai4i/ai4i.zip" \
  "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
echo "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2  $DEST/cmapss/turbofan.zip" | sha256sum -c -
echo "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252  $DEST/ai4i/ai4i.zip" | sha256sum -c -
cd "$DEST/cmapss" && unzip -q -o turbofan.zip && unzip -q -o "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" -d .
cd ../ai4i && unzip -q -o ai4i.zip
echo "datasets ready in $DEST"
