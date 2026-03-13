import os
import json
import requests
import logging
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Planet API key and endpoint
planet_api_key = 
API_ORDER_ENDPOINT = 'https://api.planet.com/compute/ops/orders/v2'
headers = {
    'Authorization': f'api-key {planet_api_key}',
    'Content-Type': 'application/json'
}

# Maximum images per order
MAX_IMAGES_PER_ORDER = 100

# Valid product bundles for PSScene
PREFERRED_PRODUCT_BUNDLES = [
    "analytic_8b_sr_udm2",      # First choice (surface reflectance,ortho)
    "analytic_8b_udm2",      	# Second choice (8-band imagery, ortho)
    "basic_analytic_8b_udm2",   	# Third choice (8-band imagery)
    ]


def load_final_images(input_dir):
    """Load the final list of selected images from the final payload file."""
    final_payload_path = os.path.join(input_dir, "planet_order_payload_final.json")

    if not os.path.exists(final_payload_path):
        logging.error(f"Final payload file {final_payload_path} not found.")
        return []

    with open(final_payload_path, 'r') as f:
        data = json.load(f)
        images = []
        for product in data.get('products', []):
            images.extend(product.get('item_ids', []))

    logging.info(f"Loaded {len(images)} images from final payload.")
    return images


def submit_order_to_planet(order_payload):
    """Submit a single order to the Planet API and handle detailed logging."""
    logging.info(f"Submitting order payload: {json.dumps(order_payload, indent=4)}")
    try:
        response = requests.post(API_ORDER_ENDPOINT, headers=headers, json=order_payload)
        response.raise_for_status()  # Raise exception if the request was unsuccessful
        order_id = response.json().get('id')
        logging.info(f"Successfully submitted order with ID: {order_id}")
        return order_id
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {e}")
        if response.status_code == 400:
            logging.error(f"400 Error Details: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to submit order: {e}")
        return None


def process_final_images_in_batches(image_list, input_dir):
    """Process and submit orders in batches of 100 images."""
    total_images = len(image_list)
    num_batches = (total_images + MAX_IMAGES_PER_ORDER - 1) // MAX_IMAGES_PER_ORDER

    for batch_num in range(num_batches):
        start_idx = batch_num * MAX_IMAGES_PER_ORDER
        end_idx = min(start_idx + MAX_IMAGES_PER_ORDER, total_images)
        batch_images = image_list[start_idx:end_idx]

        for product_bundle in PREFERRED_PRODUCT_BUNDLES:
            order_payload = {
                "name": f"Planet_Order_Batch_{batch_num + 1}",
                "products": [
                    {
                        "item_ids": batch_images,
                        "item_type": "PSScene",
                        "product_bundle": product_bundle
                    }
                ],
                "delivery": {
                    "archive_type": "zip"  # Request the results in ZIP format
                }
            }

            # Submit the order to Planet API
            order_id = submit_order_to_planet(order_payload)

            # If the order is successfully submitted, save and move to the next batch
            if order_id:
                batch_filename = f"NZM_2022_batch_{batch_num + 1}_{product_bundle}.json"
                order_payload_path = os.path.join(input_dir, batch_filename)
                with open(order_payload_path, "w") as outfile:
                    json.dump(order_payload, outfile, indent=4)
                logging.info(f"Saved order payload for batch {batch_num + 1} with bundle {product_bundle} to {order_payload_path}")
                break  # Move to the next batch if successfully ordered with a bundle

        # Add delay to avoid hitting API rate limits
        time.sleep(2)


def main():
    # Your specified input directory
    input_dir = "/data/BHCEP/Wetlands/Taranaki_Wetlands/Taranaki_Planet_data/lists/Taranaki_2020_Sep_list"
    input_dir = "/data/BHCEP/Wetlands/NZ_Wetlands/NZ_Planet_data/lists/NZ_2020_March_list"

    # Step 1: Load final list of images from the final output file
    final_images = load_final_images(input_dir)

    # Step 2: Process and submit orders in batches of 100 images
    if final_images:
        process_final_images_in_batches(final_images, input_dir)
    else:
        logging.info("No images found for ordering.")


if __name__ == "__main__":
    main()

