import createClient from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import https from 'https';
import { IncomingMessage } from 'http';
import { AtpAgent } from '@atproto/api';

dotenv.config();

const token = process.env["KEY_GITHUB_TOKEN"];
const endpoint = "https://models.inference.ai.azure.com";
const modelName = "gpt-4o";
const todayFolder = 'today';
const imageFileName = 'todays_image.jpg';
const imageUrl = 'https://webkamera.atlas.vegvesen.no/public/kamera?id=0229009_1';
const bskyHandle = process.env["BSKY_HANDLE"];
const bskyPassword = process.env["BSKY_PASSWORD"];

async function downloadImage(url: string, dest: string): Promise<void> {
    return new Promise((resolve, reject) => {
        https.get(url, (response) => {
            if (response.statusCode === 302 && response.headers.location) {
                https.get(response.headers.location, (redirectedResponse) => {
                    saveImageToFile(redirectedResponse, dest, resolve, reject);
                }).on('error', reject);
            } else if (response.statusCode === 200) {
                saveImageToFile(response, dest, resolve, reject);
            } else {
                reject(new Error(`Failed to download image. Status Code: ${response.statusCode}`));
            }
        }).on('error', reject);
    });
}

function saveImageToFile(response: IncomingMessage, dest: string, resolve: () => void, reject: (err: Error) => void): void {
    const file = fs.createWriteStream(dest);
    response.pipe(file);
    file.on('finish', () => {
        file.close(resolve);
    });
    file.on('error', (err) => {
        fs.unlink(dest, () => reject(err));
    });
}

async function getImageDescription(): Promise<string> {
    if (!token) {
        throw new Error("Azure API token is not defined");
    }

    const client = createClient(endpoint, new AzureKeyCredential(token));
    const imageDataUrl = getImageDataUrl(path.join(todayFolder, imageFileName), 'jpg');

    let response;
    try {
        response = await client.path("/chat/completions").post({
            body: {
                messages: [
                    { role: "system", content: "You are a helpful assistant that describes images in detail." },
                    {
                        role: "user", content: [
                            { type: "text", text: "Describe this image in less than 200 characters:" },
                            { type: "image_url", image_url: { url: imageDataUrl, detail: "low" } }
                        ]
                    }
                ],
                model: modelName
            }
        });
    } catch (err) {
        console.error("Azure API request failed:", err);
        throw new Error("Azure API request failed");
    }

    if (response.status !== "200") {
        console.error("Azure API error response:", response);
        if ('error' in response && typeof response.error === 'object' && response.error !== null && 'message' in response.error) {
            throw new Error((response.error as { message: string }).message);
        } else {
            throw new Error("Azure API returned non-200 status: " + response.status);
        }
    }

    if ('choices' in response.body && response.body.choices[0].message.content) {
        return response.body.choices[0].message.content;
    } else {
        console.error("Unexpected Azure API response format:", response.body);
        throw new Error("Unexpected response format");
    }
    // fallback
    return "No description available.";
}

function getImageDataUrl(imageFile: string, imageFormat: string): string {
    try {
        const imageBuffer = fs.readFileSync(imageFile);
        const imageBase64 = imageBuffer.toString('base64');
        return `data:image/${imageFormat};base64,${imageBase64}`;
    } catch {
        console.error(`Could not read '${imageFile}'.`);
        process.exit(1);
    }
}

function convertDataURIToUint8Array(dataURI: string): Uint8Array {
    const byteString = atob(dataURI.split(',')[1]);
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uint8Array = new Uint8Array(arrayBuffer);
    for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i);
    }
    return uint8Array;
}

async function getPostText(): Promise<string> {
    // Ensure the 'today' folder exists
    if (!fs.existsSync(todayFolder)) {
        fs.mkdirSync(todayFolder);
    }

    if (!bskyHandle || !bskyPassword) {
        throw new Error("Bluesky handle or password is not defined");
    }

    // Step 1: Download the image to the 'today' folder
    const imageDest = path.join(todayFolder, imageFileName);
    try {
        await downloadImage(imageUrl, imageDest);
        console.log(`Image downloaded to: ${imageDest}`);
    } catch {
        console.error("Error downloading the image.");
        return "Error downloading the image.";
    }

    // Step 2: Get image description from GPT-4o
    let imageDescription: string;
    try {
        imageDescription = await getImageDescription() || "No description available.";
        console.log("Image description received: ", imageDescription);
    } catch {
        console.error("Error getting image description.");
        return "Error getting image description.";
    }

    // Truncate the description to 300 graphemes
    const maxGraphemes = 300;
    if (imageDescription.length > maxGraphemes) {
        imageDescription = imageDescription.slice(0, maxGraphemes) + '...';
    }

    // Step 3: Upload the image as a blob to Bluesky
    const agent = new AtpAgent({ service: 'https://bsky.social' });
    await agent.login({ identifier: bskyHandle, password: bskyPassword });

    const imageDataUrl = getImageDataUrl(imageDest, 'jpg');
    const { data } = await agent.uploadBlob(convertDataURIToUint8Array(imageDataUrl), { encoding: 'image/jpeg' });

    // Step 4: Create a post with the image embedded
    await agent.post({
        text: imageDescription,
        embed: {
            $type: 'app.bsky.embed.images',
            images: [
                {
                    alt: imageDescription,
                    image: data.blob,
                    aspectRatio: {
                        width: 1000,
                        height: 500
                    }
                }
            ]
        },
        createdAt: new Date().toISOString()
    });

    return `${imageDescription}\n![Image](${imageUrl})`;
}

async function main() {
    try {
        const text = await getPostText();
        console.log(`[${new Date().toISOString()}] Posted: "${text}"`);
    } catch (error) {
        console.error("Error posting to Bluesky:", error);
    }
}

main();