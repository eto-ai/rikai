/*
 * Copyright 2021 Rikai Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.mlflow.tracking;
import org.mlflow.tracking.creds.*;

import java.net.URI;


/**
 * Client to an MLflow Tracking Sever.
 * Originally from mlflow client 1.21.0
 */
public class RikaiMlflowClient extends MlflowClient {
    private final RiMlflowHttpCaller deleteHttpCaller;

    public RikaiMlflowClient(String trackingUri) {
        this(getHostCredsProviderFromTrackingUri(trackingUri));
    }

    public RikaiMlflowClient(MlflowHostCredsProvider hostCredsProvider) {
        super(hostCredsProvider);
        this.deleteHttpCaller = new RiMlflowHttpCaller(hostCredsProvider);
    }

    /**
     * Send a DELETE to the following path, with a String-encoded JSON body.
     * This is mostly an internal API, but allows making lower-level or unsupported requests.
     * @return JSON response from the server.
     */
    public String sendDelete(String path, String json) {
        return deleteHttpCaller.delete(path, json);
    }

    private static MlflowHostCredsProvider getHostCredsProviderFromTrackingUri(String trackingUri) {
        URI uri = URI.create(trackingUri);
        MlflowHostCredsProvider provider;

        if ("http".equals(uri.getScheme()) || "https".equals(uri.getScheme())) {
            provider = new BasicMlflowHostCreds(trackingUri);
        } else if (trackingUri.equals("databricks")) {
            MlflowHostCredsProvider profileProvider = new DatabricksConfigHostCredsProvider();
            MlflowHostCredsProvider dynamicProvider =
                    DatabricksDynamicHostCredsProvider.createIfAvailable();
            if (dynamicProvider != null) {
                provider = new HostCredsProviderChain(dynamicProvider, profileProvider);
            } else {
                provider = profileProvider;
            }
        } else if ("databricks".equals(uri.getScheme())) {
            provider = new DatabricksConfigHostCredsProvider(uri.getHost());
        } else if (uri.getScheme() == null || "file".equals(uri.getScheme())) {
            throw new IllegalArgumentException("Java Client currently does not support" +
                    " local tracking URIs. Please point to a Tracking Server.");
        } else {
            throw new IllegalArgumentException("Invalid tracking server uri: " + trackingUri);
        }
        return provider;
    }
}
