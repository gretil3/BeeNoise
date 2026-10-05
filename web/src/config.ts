// config.yaml is bundled at build time: the browser app shares the Python defaults.
import { parse } from "yaml";
import raw from "../../config.yaml?raw";

export const CFG: any = parse(raw);
